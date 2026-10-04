from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import date, timedelta
from hashlib import sha256
from time import perf_counter
from types import SimpleNamespace
from typing import Literal
from urllib.parse import urlparse
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.connectors.ports import (
    JudgeChoiceAnswer,
    JudgeChoiceQuestion,
    JudgeEvaluationRequest,
)
from app.deployments import DeploymentCatalog
from app.discovery import (
    DiscoveryCandidate,
    DiscoveryContext,
    LeadQualification,
    QualificationClaim,
    QualificationGeneration,
    RejectedDiscoveryCandidate,
    SearchProviderResolver,
    SearchResult,
    qualify_search_result,
    unsupported_numeric_claims,
)
from app.domain import JsonObject, RunEventType, RunStatus, utc_now
from app.execution_controls import BudgetExceeded, bounded_service
from app.judge_policy import (
    ASSIST_REJECT_CONFIDENCE,
    AUTONOMOUS_ADVANCE_CONFIDENCE,
    AUTONOMOUS_REJECT_CONFIDENCE,
    LEAD_DECISION_POLICY_VERSION,
    JudgeEvaluator,
    JudgeMode,
    answer_confidence,
    evaluate_judgment,
)
from app.knowledge_memory import (
    KnowledgeDocument,
    KnowledgeMemoryStore,
    approved_agent_memory_context,
)
from app.market_research import MarketBrief, MarketLane, MarketResearchService, search_keywords
from app.model_provider import (
    GenerationRequest,
    ModelOutputTruncatedError,
    ModelProvider,
    ModelProviderError,
)
from app.persistence import RecordConflictError, Repository
from app.prospects import HumanReviewDecision, Prospect, ProspectStore
from app.validation import ContractValidationError, parse_and_validate_object
from app.website_evidence import WebsiteEvidenceCollector, WebsiteEvidenceError

_QUALIFICATION_MAX_NEW_TOKENS = 768
_QUALIFICATION_MAX_EVIDENCE = 4
_QUALIFICATION_RETRY_MAX_EVIDENCE = 2
_QUALIFICATION_EVIDENCE_MAX_CHARS = 720
_QUALIFICATION_RETRY_EVIDENCE_MAX_CHARS = 440
_QUALIFICATION_MEMORY_MAX_CHARS = 900
_QUALIFICATION_INSTRUCTION_MAX_CHARS = 600
_PLANNER_MAX_NEW_TOKENS = 1_024
_PLANNER_RETRY_MAX_NEW_TOKENS = 768
_PLANNER_DOCUMENT_MAX_CHARS = 1_200
_PLANNER_RETRY_DOCUMENT_MAX_CHARS = 750
_PLANNER_MAX_TRACKED = 24
_PLANNER_RETRY_MAX_TRACKED = 12
_PLANNER_MAX_MEMORY_ITEMS = 4
_PLANNER_RETRY_MAX_MEMORY_ITEMS = 2

RESEARCH_AGENT_ID = "lead.research"
_SITE_OPERATOR = re.compile(r"(?<!\S)-?site:\S+", re.IGNORECASE)
_MAX_SEARCH_QUERY_LENGTH = 450
_AUTONOMY_MIN_COMPLETED_RUNS = 2
_AUTONOMY_MIN_REVIEWED_PROSPECTS = 6
_AUTONOMY_MIN_APPROVAL_RATE = 0.85
_AUTONOMY_MAX_ARCHIVE_RATE = 0.15
_AUTONOMY_CONFIDENCE_THRESHOLD = 85
_AUTONOMY_REVIEW_WINDOW_DAYS = 30
_DISCOVERY_CONTEXT_MAX_CHARS = 1_900
_COMPANY_NAME_NOISE = frozenset(
    {
        "cia",
        "company",
        "corp",
        "corporation",
        "grupo",
        "group",
        "holding",
        "holdings",
        "inc",
        "configured country",
        "del",
        "de",
        "estate",
        "international",
        "inmobiliaria",
        "inmobiliario",
        "ltda",
        "llc",
        "sa",
        "sas",
        "service",
        "services",
        "sociedad",
        "anonima",
        "anónima",
        "the",
        "configured region",
        "real",
        "y",
    }
)
_COMPANY_NAME_TOKEN = re.compile(r"[^\W_]{3,}", re.UNICODE)


def _bounded_text(value: object, max_chars: int) -> str:
    """Normalize and cap context fields before they enter a local-model prompt."""
    return " ".join(str(value or "").split())[:max_chars]


def _bounded_text_with_tail(value: object, max_chars: int) -> str:
    """Keep both the lead and trailing contact details from a long evidence excerpt."""
    normalized = " ".join(str(value or "").split())
    if len(normalized) <= max_chars:
        return normalized
    head = max_chars * 2 // 3
    tail = max_chars - head
    return f"{normalized[:head].rstrip()} … {normalized[-tail:].lstrip()}"


def _compact_discovery_context(
    content: str, *, max_chars: int = _DISCOVERY_CONTEXT_MAX_CHARS
) -> str:
    """Keep the offer summary and its discovery hypotheses in the planning prompt.

    Discovery needs the client's targeting intent, not every paragraph in a long
    collateral document. A prefix alone can hide the hypotheses near the middle
    of the document, so retain small windows around stable, human-authored
    headings while keeping the local model prompt bounded.
    """

    normalized = " ".join(content.split())
    if len(normalized) <= max_chars:
        return normalized

    lower = normalized.casefold()
    segments: list[str] = [normalized[:500]]
    markers = (
        ("initial configured region discovery hypotheses", 1_100),
        ("fit signals for the configured region discovery workflow", 500),
        ("core offer", 400),
        ("value hypotheses to test", 400),
    )
    for marker, window in markers:
        index = lower.find(marker)
        if index < 0:
            continue
        start = max(0, index - 80)
        end = min(len(normalized), index + window)
        snippet = normalized[start:end]
        if snippet not in segments:
            segments.append(snippet)

    compact = " ".join(segments)
    return compact[:max_chars]


def _company_name_matches(company_name: str, text: str) -> bool:
    """Match a public-brand result without trusting legal suffixes or generic words.

    Registry seeds often use a legal name while a public site drops ``S.A.S.`` or ``Cia. Ltda.``.
    Require the longest meaningful token plus one other anchor when available; this keeps a
    registry page for a different company from being accepted merely because it shares a host.
    """
    tokens = [
        token
        for token in _COMPANY_NAME_TOKEN.findall(company_name.casefold())
        if token not in _COMPANY_NAME_NOISE and not token.isdigit()
    ]
    if not tokens:
        return False
    haystack = text.casefold()
    hits = {token for token in tokens if re.search(rf"(?<!\w){re.escape(token)}(?!\w)", haystack)}
    if len(tokens) == 1:
        return len(hits) == 1
    longest = max(tokens, key=len)
    return longest in hits and len(hits) >= 2


def _unsupported_workflow_fit_reason(statement: str) -> str | None:
    """Flag a qualification that explicitly says the target workflow is not evidenced."""
    normalized = " ".join(statement.casefold().split())
    markers = (
        "workflow fit is not supported",
        "workflow fit is not confirmed",
        "workflow fit remains unverified",
        "remains unverified",
        "cannot be validated",
        "not directly supported by public",
        "not supported by public",
        "does not support the workflow",
        "does not confirm",
        "does not contain any visible",
        "does not indicate any operational",
        "does not directly align",
        "not a direct target",
        "not directly supported",
        "not a direct match",
    )
    for marker in markers:
        if marker in normalized:
            return "The qualification did not find public evidence for the selected workflow."
    return None


class DiscoveryAutopilotRequest(BaseModel):
    deployment_id: str = Field(min_length=1, max_length=80)
    judge_mode: JudgeMode = JudgeMode.OFF
    include_market_research: bool = False
    # A company-scoped CRM workflow consumes only that Company's explicitly
    # approved market profile, rather than blending unrelated workspace memory.
    company_id: str | None = Field(default=None, min_length=1, max_length=200)
    # Company Brain discovery requires the profile for the active baseline revision.
    company_brain_revision: int | None = Field(default=None, ge=0)
    request_id: str | None = Field(default=None, min_length=1, max_length=80)
    max_results_per_query: int = Field(default=6, ge=1, le=8)
    max_qualified_leads: int = Field(default=6, ge=1, le=8)
    search_pages: int = Field(default=2, ge=1, le=3)


class DiscoveryLane(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sector: str = Field(min_length=2, max_length=120)
    workflow: str = Field(min_length=12, max_length=300)
    value_hypothesis: str = Field(min_length=20, max_length=600)
    # The provider query is bounded after the model selects its intent. Keeping
    # a little room here avoids rejecting a useful, specific lane before the
    # application adds its company-site search term and exclusions.
    search_query: str = Field(min_length=12, max_length=450)
    mode: Literal["validated", "exploratory"]

    @field_validator("value_hypothesis")
    @classmethod
    def keeps_hypotheses_qualitative(cls, value: str) -> str:
        """A discovery plan has no lead evidence yet, so avoid invented precision."""
        if re.search(r"\b\d+(?:[.,]\d+)?\s*%?", value):
            raise ValueError(
                "Discovery value hypotheses must stay qualitative until lead evidence is collected."
            )
        return value


class TargetPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ideal_customer_profile: str = Field(min_length=20, max_length=1_500)
    value_proposition: str = Field(min_length=20, max_length=1_500)
    exclusion_signals: list[str] = Field(min_length=1, max_length=8)
    search_lanes: list[DiscoveryLane] = Field(min_length=1, max_length=5)
    source_document_ids: list[str] = Field(min_length=1, max_length=12)
    market_profile: JsonObject | None = None

    @model_validator(mode="after")
    def requires_diverse_exploration(self) -> TargetPlan:
        sectors = {lane.sector.strip().casefold() for lane in self.search_lanes}
        if len(sectors) != len(self.search_lanes):
            raise ValueError("Discovery lanes must cover distinct sectors.")
        exploratory = [lane for lane in self.search_lanes if lane.mode == "exploratory"]
        if not exploratory:
            raise ValueError("A plan needs at least one evidence-backed exploratory lane.")
        return self


def market_target_plan(
    brief: MarketBrief,
    lanes: list[MarketLane],
    source_document_ids: list[str],
) -> TargetPlan | None:
    """Build discovery lanes only from cited model-produced findings."""
    if brief.status == "synthesis_failed":
        return None

    priorities = {finding.sector: finding.priority for finding in brief.findings}
    by_sector = {finding.sector: finding for finding in brief.findings}
    # Findings may intentionally omit unsupported hypotheses. Build discovery
    # only from cited findings; one useful lane is better than forcing a weak one.
    evidence_lanes = [
        lane for lane in lanes if lane.sector in by_sector and by_sector[lane.sector].citation_ids
    ]
    if not evidence_lanes:
        return None
    # A handoff is a projection of final model-authored interventions, not a
    # revival of preliminary guesses. Incompatible content cannot be repaired here.
    if any(
        not 20 <= len(by_sector[lane.sector].opportunity) <= 300
        or re.search(r"\d", by_sector[lane.sector].opportunity)
        for lane in evidence_lanes
    ):
        return None
    ordered_lanes = sorted(
        evidence_lanes,
        key=lambda lane: priorities.get(lane.sector, len(lanes) + 1),
    )
    profile = brief.company_profile
    ideal_customer_profile = profile.ideal_customer_profile.strip()
    value_proposition = profile.positioning.strip()
    if len(ideal_customer_profile) < 20 or len(value_proposition) < 20:
        return None
    return TargetPlan(
        ideal_customer_profile=ideal_customer_profile,
        value_proposition=value_proposition,
        exclusion_signals=["No public evidence-backed workflow fit."],
        search_lanes=[
            DiscoveryLane(
                sector=lane.sector,
                workflow=by_sector[lane.sector].opportunity,
                value_hypothesis=by_sector[lane.sector].opportunity,
                search_query=" ".join(
                    dict.fromkeys(
                        [
                            *search_keywords(lane.sector, limit=6),
                            *search_keywords(by_sector[lane.sector].opportunity, limit=10),
                            "company",
                            "website",
                        ]
                    )
                )[:240],
                # A cited workflow example is not verified client need or demand.
                mode="exploratory",
            )
            for lane in ordered_lanes
        ],
        source_document_ids=source_document_ids,
        market_profile=profile.model_dump(mode="json"),
    )


class GeneratedLeadQualification(BaseModel):
    """Structured review brief produced from one company's bounded public evidence."""

    model_config = ConfigDict(extra="forbid")

    entity_type: Literal[
        "actual_company",
        "listing_or_directory",
        "ranking_or_comparison",
        "software_or_provider",
        "unknown",
    ]
    company_name: str | None = Field(default=None, min_length=2, max_length=200)
    entity_reason: QualificationClaim
    rejection_reason: str | None = Field(default=None, min_length=10, max_length=800)
    summary: str | None = Field(default=None, min_length=20, max_length=1_200)
    company_signals: list[QualificationClaim] | None = Field(
        default=None, min_length=1, max_length=6
    )
    workflow_fit: QualificationClaim | None = None
    value_hypothesis: str | None = Field(default=None, min_length=20, max_length=1_200)
    known_unknowns: list[str] = Field(default_factory=list, max_length=8)
    confidence: int | None = Field(default=None, ge=0, le=100)

    @model_validator(mode="after")
    def requires_a_complete_brief_only_for_actual_companies(self) -> GeneratedLeadQualification:
        if self.entity_type == "actual_company":
            missing = [
                name
                for name, value in (
                    ("summary", self.summary),
                    ("company_signals", self.company_signals),
                    ("workflow_fit", self.workflow_fit),
                    ("value_hypothesis", self.value_hypothesis),
                    ("confidence", self.confidence),
                )
                if value is None or value == []
            ]
            if missing:
                raise ValueError(
                    "An actual company requires these qualification fields: "
                    + ", ".join(missing)
                    + "."
                )
            if self.rejection_reason:
                raise ValueError("An actual company cannot also have a rejection reason.")
        elif not self.rejection_reason:
            raise ValueError("A non-company result requires a rejection reason.")
        return self


class DiscoveryAutonomyPolicy(BaseModel):
    """M8-only shadow gate derived exclusively from attributable human reviews."""

    gate_met: bool
    review_window_days: int = Field(default=_AUTONOMY_REVIEW_WINDOW_DAYS, ge=1)
    completed_discovery_runs: int = Field(ge=0)
    reviewed_prospects: int = Field(ge=0)
    approved_prospects: int = Field(ge=0)
    archived_prospects: int = Field(ge=0)
    approval_rate: float | None = Field(default=None, ge=0, le=1)
    archive_rate: float | None = Field(default=None, ge=0, le=1)
    confidence_threshold: int = Field(default=_AUTONOMY_CONFIDENCE_THRESHOLD, ge=0, le=100)
    reason: str = Field(min_length=1, max_length=300)


class DiscoveryLaneReviewOutcome(BaseModel):
    sector: str = Field(min_length=1, max_length=160)
    mode: Literal["validated", "exploratory"]
    reviewed_prospects: int = Field(ge=0)
    approved_prospects: int = Field(ge=0)
    archived_prospects: int = Field(ge=0)
    approval_rate: float | None = Field(default=None, ge=0, le=1)


class JudgeLeadDecision(BaseModel):
    candidate_name: str = Field(min_length=1, max_length=200)
    choice: Literal["advance", "reject", "review"] | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    disposition: Literal[
        "observed",
        "filtered",
        "auto_declined",
        "pending_review",
        "auto_advanced",
    ]
    rule_version: str = LEAD_DECISION_POLICY_VERSION
    prospect_id: str | None = None


class DiscoveryAutopilotResponse(BaseModel):
    plan: TargetPlan
    market_brief: MarketBrief | None = None
    candidates: list[DiscoveryCandidate]
    queued_prospects: list[Prospect] = Field(default_factory=list)
    rejected_candidates: list[RejectedDiscoveryCandidate] = Field(default_factory=list)
    sources_examined: int = Field(ge=0)
    existing_prospects_skipped: int = Field(default=0, ge=0)
    autonomy: DiscoveryAutonomyPolicy
    lane_review_outcomes: list[DiscoveryLaneReviewOutcome] = Field(default_factory=list)
    shadow_candidate_prospect_ids: list[str] = Field(default_factory=list, max_length=8)
    judge_decisions: list[JudgeLeadDecision] = Field(default_factory=list, max_length=8)
    auto_advanced_prospect_ids: list[str] = Field(default_factory=list, max_length=8)


class DiscoveryAutopilotError(RuntimeError):
    pass


class DiscoveryAutopilotService:
    def __init__(
        self,
        deployments: DeploymentCatalog,
        knowledge: KnowledgeMemoryStore,
        provider: ModelProvider,
        search: SearchProviderResolver,
        prospects: ProspectStore,
        validation_attempts: int,
        website_evidence: WebsiteEvidenceCollector | None = None,
        repository: Repository | None = None,
        judge_evaluator: JudgeEvaluator | None = None,
    ) -> None:
        self._deployments = deployments
        self._knowledge = knowledge
        self._provider = provider
        self._search = search
        self._prospects = prospects
        self._validation_attempts = validation_attempts
        self._website_evidence = website_evidence
        self._repository = repository
        self._judge_evaluator = judge_evaluator

    @bounded_service
    def run(
        self,
        request: DiscoveryAutopilotRequest,
        record: Callable[[RunEventType, JsonObject], None] | None = None,
    ) -> DiscoveryAutopilotResponse:
        deployment = self._deployments.get(request.deployment_id)
        overlay = next(
            (
                item
                for item in deployment.agent_overlays
                if item.agent_id == RESEARCH_AGENT_ID and item.enabled
            ),
            None,
        )
        if overlay is None:
            raise RecordConflictError(
                "The selected deployment has no enabled lead.research overlay."
            )
        if not set(overlay.allowed_knowledge_collections).issubset(
            deployment.knowledge_collections
        ):
            raise DiscoveryAutopilotError(
                "Discovery may only read collections declared by its deployment."
            )
        documents = (
            self._knowledge.list_documents(
                deployment.deployment_id, overlay.allowed_knowledge_collections
            )
            if overlay.allowed_knowledge_collections
            else []
        )
        existing_prospects = self._prospects.list()
        autonomy = self._autonomy_policy(deployment.deployment_id, existing_prospects)
        existing_prospect_ids = {prospect.prospect_id for prospect in existing_prospects}
        known_hosts = {
            self._source_host(source.source_url)
            for prospect in existing_prospects
            for source in prospect.sources
            if source.source_type != "public_company_registry"
            if self._source_host(source.source_url)
        }
        known_names = {
            self._company_key(prospect.company_name)
            for prospect in existing_prospects
            if self._company_key(prospect.company_name)
        }
        approved_memory = approved_agent_memory_context(
            self._knowledge,
            deployment.deployment_id,
            "lead.discovery",
            query="sector workflow qualification discovery",
            entity_ids=[request.company_id] if request.company_id else None,
        )
        if request.company_id and not documents:
            company_profile_memory = next(
                (
                    item
                    for item in approved_memory.get("semantic", [])
                    if item.get("predicate") == "ayni.company.market_profile"
                ),
                None,
            )
            profile_value = (
                company_profile_memory.get("value")
                if isinstance(company_profile_memory, dict)
                else None
            )
            profile_run_id = (
                profile_value.get("market_research_run_id")
                if isinstance(profile_value, dict)
                else None
            )
            if isinstance(profile_run_id, str):
                documents = [
                    SimpleNamespace(
                        document_id=f"crm-market-profile:{profile_run_id}",
                        title="Approved company market profile",
                        content=json.dumps(profile_value, ensure_ascii=False),
                    )
                ]
        if not documents:
            raise DiscoveryAutopilotError(
                "No approved company context exists. Add company knowledge or approve a "
                "Company market profile before lead discovery."
            )
        client_context = None
        market_brief = None
        if request.include_market_research:
            client_context = self._market_client_context(
                deployment.deployment_id,
                deployment.company_name or deployment.display_name,
                documents,
            )
            market_brief = self._saved_client_market_brief(client_context)
        if market_brief is not None:
            if not market_brief.target_plan:
                raise DiscoveryAutopilotError(
                    "Market research did not produce a model-built target plan. "
                    "Review the company profile before discovering leads."
                )
            plan = TargetPlan.model_validate(market_brief.target_plan)
            if record:
                invocation_id = str(uuid4())
                envelope = {"invocation_id": invocation_id, "tool_name": "market_research"}
                record(
                    RunEventType.TOOL_REQUESTED,
                    {**envelope, "arguments": {"deployment_id": deployment.deployment_id}},
                )
                record(
                    RunEventType.TOOL_COMPLETED,
                    {**envelope, "reused": True, "result": market_brief.model_dump(mode="json")},
                )
        elif request.include_market_research:
            market_brief = MarketResearchService(
                self._provider, self._validation_attempts
            ).research(
                None,
                lambda query: self._search_results(deployment.deployment_id, query, 3, record),
                record,
                client_context=client_context,
                website_evidence=(
                    self._website_evidence.collect if self._website_evidence is not None else None
                ),
            )
            plan = market_target_plan(
                market_brief,
                market_brief.hypotheses,
                [str(document.document_id) for document in documents],
            )
            if plan is None:
                raise DiscoveryAutopilotError(
                    "Market research did not produce a model-built target plan. "
                    "Review the company profile before discovering leads."
                )
            plan.market_profile = market_brief.company_profile.model_dump(mode="json")
            market_brief.target_plan = plan.model_dump(mode="json")
        else:
            plan = self._plan(
                deployment.deployment_id,
                documents,
                existing_prospects,
                record,
                approved_memory,
            )
        known_ids = {str(document.document_id) for document in documents}
        if not set(plan.source_document_ids).issubset(known_ids):
            raise DiscoveryAutopilotError("The target plan referenced knowledge it was not given.")
        candidates: list[DiscoveryCandidate] = []
        rejected_candidates: list[RejectedDiscoveryCandidate] = []
        seen_urls: set[str] = set()
        seen_hosts: set[str] = set()
        existing_prospects_skipped = 0
        sources_examined = 0
        for lane in plan.search_lanes:
            company_query = self._company_first_query(
                lane.search_query,
                excluded_hosts=known_hosts,
                cycle_date=utc_now().date() if existing_prospects else None,
            )
            for page in range(request.search_pages):
                results = self._search_results(
                    deployment.deployment_id,
                    company_query,
                    request.max_results_per_query,
                    record,
                    offset=page,
                )
                sources_examined += len(results)
                for result in results:
                    if result.url in seen_urls:
                        continue
                    seen_urls.add(result.url)
                    result_host = self._source_host(result.url)
                    deduplicates_by_host = result.source_type != "public_company_registry"
                    if deduplicates_by_host and result_host in seen_hosts:
                        continue
                    if result_host in known_hosts:
                        existing_prospects_skipped += 1
                        rejected_candidates.append(
                            RejectedDiscoveryCandidate(
                                source_url=result.url,
                                source_title=result.title,
                                source_excerpt=result.description,
                                search_query=company_query,
                                rejection_reasons=[
                                    "This company is already tracked in the workspace; it was "
                                    "skipped so discovery can return a new lead."
                                ],
                            )
                        )
                        continue
                    qualified = qualify_search_result(result, company_query)
                    if isinstance(qualified, DiscoveryCandidate):
                        if self._company_key(qualified.company_name) in known_names:
                            existing_prospects_skipped += 1
                            rejected_candidates.append(
                                RejectedDiscoveryCandidate(
                                    source_url=result.url,
                                    source_title=result.title,
                                    source_excerpt=result.description,
                                    search_query=company_query,
                                    rejection_reasons=[
                                        "This company name is already tracked in the workspace; it "
                                        "was skipped so discovery can return a new lead."
                                    ],
                                )
                            )
                            continue
                        if deduplicates_by_host and result_host:
                            seen_hosts.add(result_host)
                        candidates.append(
                            qualified.model_copy(
                                update={
                                    "discovery_context": DiscoveryContext(
                                        sector=lane.sector,
                                        workflow=lane.workflow,
                                        value_hypothesis=lane.value_hypothesis,
                                        mode=lane.mode,
                                        search_query=company_query,
                                        company_id=request.company_id,
                                    )
                                }
                            )
                        )
                    else:
                        rejected_candidates.append(qualified)
        qualified_candidates: list[DiscoveryCandidate] = []
        shadow_candidate_prospect_ids: list[str] = []
        for candidate in candidates:
            candidate_key = self._company_key(candidate.company_name)
            if candidate_key and candidate_key in known_names:
                existing_prospects_skipped += 1
                rejected_candidates.append(
                    RejectedDiscoveryCandidate(
                        source_url=candidate.source_url,
                        source_title=candidate.source_title,
                        source_excerpt=candidate.source_excerpt,
                        search_query=candidate.search_query,
                        rejection_reasons=[
                            "This company appeared more than once in the discovery cycle; "
                            "the duplicate was skipped."
                        ],
                    )
                )
                continue
            try:
                outcome = self._qualify_candidate(deployment.deployment_id, candidate, record)
            except DiscoveryAutopilotError as error:
                rejected_candidates.append(
                    RejectedDiscoveryCandidate(
                        source_url=candidate.source_url,
                        source_title=candidate.source_title,
                        source_excerpt=candidate.source_excerpt,
                        search_query=candidate.search_query,
                        rejection_reasons=[
                            "Agent qualification could not produce a valid evidence-backed brief "
                            f"for this candidate: {error}"
                        ],
                    )
                )
                continue
            if isinstance(outcome, DiscoveryCandidate):
                outcome_key = self._company_key(outcome.company_name)
                if outcome_key and outcome_key in known_names:
                    existing_prospects_skipped += 1
                    rejected_candidates.append(
                        RejectedDiscoveryCandidate(
                            source_url=outcome.source_url,
                            source_title=outcome.source_title,
                            source_excerpt=outcome.source_excerpt,
                            search_query=outcome.search_query,
                            rejection_reasons=[
                                "The qualification result matches a company already retained "
                                "in this discovery cycle; the duplicate was skipped."
                            ],
                        )
                    )
                    continue
                qualified_candidates.append(outcome)
                company_key = self._company_key(outcome.company_name)
                if company_key:
                    known_names.add(company_key)
                if len(qualified_candidates) >= request.max_qualified_leads:
                    break
            else:
                rejected_candidates.append(outcome)

        judged_choices: dict[int, tuple[str, float]] = {}
        if request.judge_mode is not JudgeMode.OFF and qualified_candidates:
            judge_questions = {
                f"candidate_{index}": JudgeChoiceQuestion(
                    type="choice",
                    instructions=(
                        "Classify this evidence-backed candidate for the approved company profile. "
                        "Advance only when public evidence shows a clear fit. Reject only when "
                        "evidence shows a clear mismatch. Use review when evidence is weak, mixed, "
                        "or incomplete. Do not infer facts that are not in the supplied state."
                    ),
                    criteria={
                        "advance": "Clear evidence-backed fit for the approved profile.",
                        "reject": "Clear evidence-backed mismatch with the approved profile.",
                        "review": "Insufficient or conflicting evidence; a person should review.",
                    },
                )
                for index in range(len(qualified_candidates))
            }
            judge_state: JsonObject = {
                "company_id": request.company_id or "",
                "approved_target_profile": {
                    "ideal_customer_profile": plan.ideal_customer_profile[:1_200],
                    "value_proposition": plan.value_proposition[:1_200],
                    "exclusion_signals": [item[:240] for item in plan.exclusion_signals[:8]],
                },
                "candidates": [
                    {
                        "candidate_id": f"candidate_{index}",
                        "company_name": candidate.company_name[:160],
                        "fit_score": candidate.fit_score,
                        "qualification": (
                            {
                                "summary": candidate.qualification.summary[:600],
                                "company_signals": [
                                    {
                                        "statement": claim.statement[:280],
                                        "source_urls": [str(claim.source_urls[0])[:240]],
                                    }
                                    for claim in candidate.qualification.company_signals[:2]
                                ],
                                "workflow_fit": {
                                    "statement": (
                                        candidate.qualification.workflow_fit.statement[:350]
                                    ),
                                    "source_urls": [
                                        str(candidate.qualification.workflow_fit.source_urls[0])[
                                            :240
                                        ]
                                    ],
                                },
                                "value_hypothesis": candidate.qualification.value_hypothesis[:450],
                                "confidence": candidate.qualification.confidence,
                                "known_unknowns": [
                                    item[:120]
                                    for item in candidate.qualification.known_unknowns[:3]
                                ],
                            }
                            if candidate.qualification
                            else {}
                        ),
                        "discovery_context": (
                            {
                                "sector": candidate.discovery_context.sector[:100],
                                "workflow": candidate.discovery_context.workflow[:180],
                                "value_hypothesis": (
                                    candidate.discovery_context.value_hypothesis[:240]
                                ),
                                "mode": candidate.discovery_context.mode,
                            }
                            if candidate.discovery_context
                            else {}
                        ),
                        "source_excerpt": candidate.source_excerpt[:500],
                        "sources": [
                            {
                                "url": str(source.url)[:240],
                                "title": source.title[:180],
                                "excerpt": source.description[:450],
                            }
                            for source in candidate.public_sources[:2]
                        ],
                    }
                    for index, candidate in enumerate(qualified_candidates)
                ],
            }
            judge_result = evaluate_judgment(
                request.judge_mode,
                self._judge_evaluator,
                JudgeEvaluationRequest(state=judge_state, questions=judge_questions),
                decision_id=f"lead-routing:{request.request_id or uuid4()}",
                deployment_id=deployment.deployment_id,
                record=record,
                requested_label="Judge evaluates lead fit",
                completed_label="Judge evaluated lead candidates",
            )
            if request.judge_mode is JudgeMode.AUTONOMOUS and judge_result is None:
                raise DiscoveryAutopilotError(
                    "The Judge did not return a result. Auto mode stopped without routing leads."
                )
            if judge_result is not None:
                for index in range(len(qualified_candidates)):
                    answer = judge_result.answers.get(f"candidate_{index}")
                    if not isinstance(answer, JudgeChoiceAnswer):
                        if request.judge_mode is JudgeMode.AUTONOMOUS:
                            raise DiscoveryAutopilotError(
                                "The Judge returned an incomplete result. "
                                "Auto mode stopped without routing leads."
                            )
                        continue
                    if answer.choice not in {"advance", "reject", "review"}:
                        if request.judge_mode is JudgeMode.AUTONOMOUS:
                            raise DiscoveryAutopilotError(
                                "The Judge returned an unsupported choice. "
                                "Auto mode stopped without routing leads."
                            )
                        continue
                    judged_choices[index] = (answer.choice, answer_confidence(answer))
                if request.judge_mode is JudgeMode.AUTONOMOUS and len(judged_choices) != len(
                    qualified_candidates
                ):
                    raise DiscoveryAutopilotError(
                        "The Judge did not decide every candidate. "
                        "Auto mode stopped without routing leads."
                    )

        queued_prospects: list[Prospect] = []
        judge_decisions: list[JudgeLeadDecision] = []
        auto_advanced_prospect_ids: list[str] = []
        for index, candidate in enumerate(qualified_candidates):
            queued: Prospect | None = None
            choice_confidence = judged_choices.get(index)
            choice, confidence = choice_confidence if choice_confidence else (None, None)
            disposition = "pending_review"
            if choice is not None and request.judge_mode is JudgeMode.OBSERVE:
                disposition = "observed"
            elif (
                request.judge_mode is JudgeMode.AUTONOMOUS
                and choice == "advance"
                and confidence is not None
                and confidence >= AUTONOMOUS_ADVANCE_CONFIDENCE
            ):
                disposition = "auto_advanced"
            elif (
                request.judge_mode is JudgeMode.AUTONOMOUS
                and choice == "reject"
                and confidence is not None
                and confidence >= AUTONOMOUS_REJECT_CONFIDENCE
            ):
                disposition = "filtered"
            elif request.judge_mode is JudgeMode.AUTONOMOUS:
                disposition = "auto_declined"
            elif (
                choice == "reject"
                and confidence is not None
                and confidence >= ASSIST_REJECT_CONFIDENCE
                and request.judge_mode is JudgeMode.ASSIST
            ):
                disposition = "filtered"

            if disposition == "filtered":
                rejected_candidates.append(
                    RejectedDiscoveryCandidate(
                        source_url=candidate.source_url,
                        source_title=candidate.source_title,
                        source_excerpt=candidate.source_excerpt,
                        search_query=candidate.search_query,
                        rejection_reasons=[
                            "Judge filtered this candidate as a high-confidence mismatch "
                            "with the approved Company profile."
                        ],
                    )
                )
            elif disposition == "auto_declined":
                rejected_candidates.append(
                    RejectedDiscoveryCandidate(
                        source_url=candidate.source_url,
                        source_title=candidate.source_title,
                        source_excerpt=candidate.source_excerpt,
                        search_query=candidate.search_query,
                        rejection_reasons=[
                            "Auto mode did not find enough evidence to advance this candidate."
                        ],
                    )
                )
            else:
                queued = self._prospects.queue_discovery_candidate(candidate)
                if queued.prospect_id not in existing_prospect_ids:
                    if (
                        autonomy.gate_met
                        and queued.lead_qualification is not None
                        and queued.lead_qualification.confidence >= autonomy.confidence_threshold
                    ):
                        shadow_candidate_prospect_ids.append(str(queued.prospect_id))
                    queued_prospects.append(queued)
                if disposition == "auto_advanced":
                    if queued.prospect_id in existing_prospect_ids:
                        disposition = "pending_review"
                    else:
                        auto_advanced_prospect_ids.append(str(queued.prospect_id))

            decision = JudgeLeadDecision(
                candidate_name=candidate.company_name,
                choice=choice,
                confidence=confidence,
                disposition=disposition,
                prospect_id=str(queued.prospect_id) if queued else None,
            )
            if request.judge_mode is not JudgeMode.OFF:
                judge_decisions.append(decision)
            if record and choice is not None:
                record(
                    RunEventType.TOOL_COMPLETED,
                    {
                        "tool_name": "judge.route_candidate",
                        "decision_id": f"lead-routing:{index}",
                        "display_label": (
                            f"Judge {disposition.replace('_', ' ')} · {candidate.company_name}"
                        )[:120],
                        "mode": request.judge_mode.value,
                        "rule_version": LEAD_DECISION_POLICY_VERSION,
                        "choice": choice,
                        "confidence": confidence or 0.0,
                        "prospect_id": decision.prospect_id,
                    },
                )
        return DiscoveryAutopilotResponse(
            plan=plan,
            market_brief=market_brief,
            candidates=qualified_candidates,
            queued_prospects=queued_prospects,
            rejected_candidates=rejected_candidates,
            sources_examined=sources_examined,
            existing_prospects_skipped=existing_prospects_skipped,
            autonomy=autonomy,
            lane_review_outcomes=self._lane_review_outcomes(existing_prospects),
            shadow_candidate_prospect_ids=shadow_candidate_prospect_ids,
            judge_decisions=judge_decisions,
            auto_advanced_prospect_ids=auto_advanced_prospect_ids,
        )

    @staticmethod
    def _market_client_context(deployment_id, company_name, documents) -> JsonObject:
        ordered = sorted(documents, key=lambda document: str(document.document_id))
        source_context = [
            {"document_id": str(document.document_id), "content": document.content}
            for document in ordered
        ]
        fingerprint = sha256(
            json.dumps(
                {
                    "version": "company-market-v1",
                    "deployment_id": deployment_id,
                    "company_name": company_name,
                    "documents": source_context,
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        return {
            "deployment_id": deployment_id,
            "company_name": company_name,
            "context_fingerprint": fingerprint,
            "source_document_ids": [str(document.document_id) for document in ordered],
            "approved_offering": [
                {
                    "document_id": str(document.document_id),
                    "content": _compact_discovery_context(document.content, max_chars=650),
                }
                for document in ordered[:2]
            ],
        }

    def _saved_client_market_brief(self, client_context: JsonObject) -> MarketBrief | None:
        if self._repository is None:
            return None
        for run in self._repository.list_runs(100):
            if (
                run.agent_id not in {"lead.discovery", "market.research"}
                or run.status != RunStatus.COMPLETED
                or run.input.get("deployment_id") != client_context["deployment_id"]
                or not run.output
            ):
                continue
            try:
                brief = MarketBrief.model_validate(
                    run.output.get("market_brief")
                    if run.agent_id == "lead.discovery"
                    else run.output
                )
                TargetPlan.model_validate(brief.target_plan)
                if (
                    brief.status == "completed"
                    and brief.deployment_id == client_context["deployment_id"]
                    and brief.context_fingerprint == client_context["context_fingerprint"]
                    and utc_now() - timedelta(days=7) <= brief.created_at <= utc_now()
                ):
                    return brief
            except (ValueError, TypeError):
                continue
        return None

    def _autonomy_policy(
        self, deployment_id: str, prospects: list[Prospect]
    ) -> DiscoveryAutonomyPolicy:
        """Compute a conservative, inspectable policy from recent human decisions only."""
        cutoff = utc_now() - timedelta(days=_AUTONOMY_REVIEW_WINDOW_DAYS)
        completed_runs = 0
        if self._repository is not None:
            completed_runs = sum(
                1
                for run in self._repository.list_runs(500)
                if run.agent_id == "lead.discovery"
                and run.status is RunStatus.COMPLETED
                and run.input.get("deployment_id") == deployment_id
                and run.created_at >= cutoff
            )
        reviewed = [
            prospect
            for prospect in prospects
            if prospect.discovery_context is not None
            and prospect.human_review_decision is not None
            and prospect.human_reviewed_at is not None
            and prospect.human_reviewed_at >= cutoff
        ]
        approved = [
            prospect
            for prospect in reviewed
            if prospect.human_review_decision is HumanReviewDecision.APPROVED
        ]
        archived = [
            prospect
            for prospect in reviewed
            if prospect.human_review_decision is HumanReviewDecision.ARCHIVED
        ]
        reviewed_count = len(reviewed)
        approval_rate = len(approved) / reviewed_count if reviewed_count else None
        archive_rate = len(archived) / reviewed_count if reviewed_count else None
        gate_met = (
            completed_runs >= _AUTONOMY_MIN_COMPLETED_RUNS
            and reviewed_count >= _AUTONOMY_MIN_REVIEWED_PROSPECTS
            and approval_rate is not None
            and approval_rate >= _AUTONOMY_MIN_APPROVAL_RATE
            and archive_rate is not None
            and archive_rate <= _AUTONOMY_MAX_ARCHIVE_RATE
        )
        if gate_met:
            reason = "Shadow gate met; high-confidence leads will be flagged without advancing."
        else:
            unmet: list[str] = []
            if completed_runs < _AUTONOMY_MIN_COMPLETED_RUNS:
                unmet.append(f"{_AUTONOMY_MIN_COMPLETED_RUNS} completed discovery runs")
            if reviewed_count < _AUTONOMY_MIN_REVIEWED_PROSPECTS:
                unmet.append(f"{_AUTONOMY_MIN_REVIEWED_PROSPECTS} reviewed discovered leads")
            if approval_rate is None or approval_rate < _AUTONOMY_MIN_APPROVAL_RATE:
                unmet.append(f"{int(_AUTONOMY_MIN_APPROVAL_RATE * 100)}% approval rate")
            if archive_rate is None or archive_rate > _AUTONOMY_MAX_ARCHIVE_RATE:
                unmet.append(f"at most {int(_AUTONOMY_MAX_ARCHIVE_RATE * 100)}% archived")
            reason = "Waiting for " + ", ".join(unmet) + "."
        return DiscoveryAutonomyPolicy(
            gate_met=gate_met,
            completed_discovery_runs=completed_runs,
            reviewed_prospects=reviewed_count,
            approved_prospects=len(approved),
            archived_prospects=len(archived),
            approval_rate=approval_rate,
            archive_rate=archive_rate,
            reason=reason,
        )

    @staticmethod
    def _lane_review_outcomes(prospects: list[Prospect]) -> list[DiscoveryLaneReviewOutcome]:
        cutoff = utc_now() - timedelta(days=_AUTONOMY_REVIEW_WINDOW_DAYS)
        lanes: dict[tuple[str, str], list[Prospect]] = {}
        for prospect in prospects:
            context = prospect.discovery_context
            if (
                context is None
                or prospect.human_review_decision is None
                or prospect.human_reviewed_at is None
                or prospect.human_reviewed_at < cutoff
            ):
                continue
            lanes.setdefault((context.sector, context.mode), []).append(prospect)

        outcomes: list[DiscoveryLaneReviewOutcome] = []
        for (sector, mode), reviewed in sorted(lanes.items()):
            approved = sum(
                prospect.human_review_decision is HumanReviewDecision.APPROVED
                for prospect in reviewed
            )
            archived = sum(
                prospect.human_review_decision is HumanReviewDecision.ARCHIVED
                for prospect in reviewed
            )
            outcomes.append(
                DiscoveryLaneReviewOutcome(
                    sector=sector,
                    mode=mode,
                    reviewed_prospects=len(reviewed),
                    approved_prospects=approved,
                    archived_prospects=archived,
                    approval_rate=approved / len(reviewed),
                )
            )
        return outcomes

    @staticmethod
    def _source_host(url: object) -> str:
        return (urlparse(str(url)).hostname or "").casefold().removeprefix("www.")

    @staticmethod
    def _company_key(value: str) -> str:
        return " ".join(value.casefold().split())

    @staticmethod
    def _company_first_query(
        query: str,
        *,
        excluded_hosts: set[str] | None = None,
        cycle_date: date | None = None,
    ) -> str:
        """Keep the model's segment intent, while making each web query seek a company site."""
        # The model selects the business/workflow lens; provider-specific search operators remain
        # application-owned. Stripping generated site operators prevents accidental positive-site
        # restrictions and duplicated exclusions that Brave rejects as an invalid query.
        model_query = " ".join(_SITE_OPERATOR.sub("", query).split())
        base = f"{model_query} company website"
        # Search engines tend to return the same ranked pages for an unchanged query. A
        # small, human-readable daily angle gives repeated discovery cycles a new lens while
        # retaining the model's segment intent. The date is deliberately not sent as a nonce.
        if cycle_date is not None:
            angles = (
                "operaciones internas",
                "solicita cotización",
                "proveedores locales",
                "atención a clientes",
                "procesos administrativos",
                "servicios empresariales",
            )
            ordinal = getattr(cycle_date, "toordinal", lambda: 0)()
            base = f"{base} {angles[ordinal % len(angles)]}"
        parts = [base]
        for host in sorted(excluded_hosts or set()):
            if not host:
                continue
            exclusion = f"-site:{host}"
            if len(" ".join([*parts, exclusion])) > _MAX_SEARCH_QUERY_LENGTH:
                break
            parts.append(exclusion)
        return " ".join(parts)

    def _qualify_candidate(
        self,
        deployment_id: str,
        candidate: DiscoveryCandidate,
        record: Callable[[RunEventType, JsonObject], None] | None = None,
    ) -> DiscoveryCandidate | RejectedDiscoveryCandidate:
        """Gather a bounded company evidence set, then create a cited review brief.

        This is deliberately M8-specific: it turns discovery results into a decision-ready
        prospect without promoting sales qualification into platform core.
        """
        deployment = self._deployments.get(deployment_id)
        overlay = next(
            (
                item
                for item in deployment.agent_overlays
                if item.agent_id == RESEARCH_AGENT_ID and item.enabled
            ),
            None,
        )
        if overlay is None:
            raise RecordConflictError(
                "The selected deployment has no enabled lead.research overlay."
            )
        profile = deployment.model_profile_for(overlay)

        public_sources = self._company_sources(deployment_id, candidate, record)
        if candidate.source_type == "public_company_registry" and not any(
            source.source_type != "public_company_registry" for source in public_sources
        ):
            raise DiscoveryAutopilotError(
                "The official registry confirms the entity, but no independent public company "
                "evidence was found."
            )
        seed_url = str(candidate.source_url).rstrip("/").casefold()
        seed_evidence = next(
            (
                source
                for source in public_sources
                if str(source.url).rstrip("/").casefold() == seed_url
            ),
            None,
        )
        if seed_evidence is not None:
            candidate = candidate.model_copy(update={"source_excerpt": seed_evidence.description})
        public_sources = [
            source
            for source in public_sources
            if str(source.url).rstrip("/").casefold() != seed_url
        ][:3]
        evidence = [
            SearchResult(
                title=candidate.source_title,
                url=candidate.source_url,
                description=candidate.source_excerpt,
                source_type=candidate.source_type,
            ),
            *public_sources,
        ]
        approved_memory = approved_agent_memory_context(
            self._knowledge,
            deployment_id,
            "lead.discovery",
            query=candidate.source_title,
        )
        generated, generation = self._generate_qualification(
            candidate,
            evidence,
            profile.model_id,
            overlay.instruction_appendix,
            record,
            approved_memory,
        )
        allowed_urls = {str(item.url) for item in evidence}
        claims = [generated.entity_reason]
        if generated.entity_type == "actual_company":
            claims.extend([*(generated.company_signals or []), generated.workflow_fit])
        cited_urls = {str(url) for claim in claims for url in claim.source_urls}
        if not cited_urls or not cited_urls.issubset(allowed_urls):
            raise DiscoveryAutopilotError(
                "The qualification brief cited public sources it was not given."
            )
        if generated.entity_type != "actual_company":
            return RejectedDiscoveryCandidate(
                source_url=candidate.source_url,
                source_title=candidate.source_title,
                source_excerpt=candidate.source_excerpt,
                search_query=candidate.search_query,
                rejection_reasons=[
                    f"Agent entity verification: {generated.rejection_reason}",
                    generated.entity_reason.statement,
                ],
            )
        evidence_text = " ".join(f"{item.title} {item.description}" for item in evidence)
        unsupported_numbers = unsupported_numeric_claims(generated.value_hypothesis, evidence_text)
        if unsupported_numbers:
            return RejectedDiscoveryCandidate(
                source_url=candidate.source_url,
                source_title=candidate.source_title,
                source_excerpt=candidate.source_excerpt,
                search_query=candidate.search_query,
                rejection_reasons=[
                    "Agent qualification contained an unsupported numeric claim: "
                    + ", ".join(unsupported_numbers),
                ],
            )
        workflow_rejection = _unsupported_workflow_fit_reason(generated.workflow_fit.statement)
        if workflow_rejection:
            return RejectedDiscoveryCandidate(
                source_url=candidate.source_url,
                source_title=candidate.source_title,
                source_excerpt=candidate.source_excerpt,
                search_query=candidate.search_query,
                rejection_reasons=[workflow_rejection],
            )
        qualification = LeadQualification(
            summary=generated.summary,
            company_signals=generated.company_signals,
            workflow_fit=generated.workflow_fit,
            value_hypothesis=generated.value_hypothesis,
            value_hypothesis_evidence_checked=True,
            known_unknowns=generated.known_unknowns,
            confidence=generated.confidence,
            generation=generation,
        )
        return candidate.model_copy(
            update={
                "company_name": generated.company_name or candidate.company_name,
                "public_sources": public_sources,
                "qualification": qualification,
            }
        )

    def _company_sources(
        self,
        deployment_id: str,
        candidate: DiscoveryCandidate,
        record: Callable[[RunEventType, JsonObject], None] | None = None,
    ) -> list[SearchResult]:
        """Retrieve a small, relevant set rather than treating all search results as evidence."""
        result_host = (
            (urlparse(str(candidate.source_url)).hostname or "").casefold().removeprefix("www.")
        )
        crawled = self._crawl_company_website(candidate, record)
        queries = [f'"{candidate.company_name}" company services website']
        if candidate.source_type == "public_company_registry":
            # Legal suffixes often prevent web indexes from matching the public brand. A second
            # unquoted query gives the registry seed a fair first-party lookup without turning the
            # discovery path into an unbounded search loop.
            public_name = re.sub(
                r"\b(?:s\.?a\.?s?\.?|cia\.?|ltda\.?|sociedad an[oó]nima|compa[nñ]ia)\b",
                " ",
                candidate.company_name,
                flags=re.IGNORECASE,
            )
            public_name = " ".join(public_name.split())
            if public_name and public_name.casefold() != candidate.company_name.casefold():
                queries.append(f"{public_name} empresa servicios sitio oficial")
        results: list[SearchResult] = []
        for query in queries:
            results.extend(self._search_results(deployment_id, query, 4, record))
        sources: list[SearchResult] = []
        seen_urls: set[str] = set()
        for result in [*crawled, *results]:
            result_host_candidate = (
                (urlparse(str(result.url)).hostname or "").casefold().removeprefix("www.")
            )
            haystack = f"{result.title} {result.description}".casefold()
            same_site = result_host_candidate == result_host
            names_company = _company_name_matches(candidate.company_name, haystack)
            normalized_url = str(result.url).rstrip("/").casefold()
            # A public-registry seed and a Brave result can share the registry host. That is not
            # independent evidence: only a matching public-brand result may enrich a registry
            # seed. Non-registry candidates may still use same-host pages as first-party evidence.
            same_site = same_site and candidate.source_type != "public_company_registry"
            if normalized_url in seen_urls or not (same_site or names_company):
                continue
            seen_urls.add(normalized_url)
            sources.append(result)
            if len(sources) == 4:
                break
        return sources

    def _crawl_company_website(
        self,
        candidate: DiscoveryCandidate,
        record: Callable[[RunEventType, JsonObject], None] | None,
    ) -> list[SearchResult]:
        if self._website_evidence is None or candidate.source_type == "public_company_registry":
            return []
        invocation_id = str(uuid4())
        if record:
            record(
                RunEventType.TOOL_REQUESTED,
                {
                    "invocation_id": invocation_id,
                    "tool_name": "website_evidence",
                    "arguments": {"url": str(candidate.source_url)},
                },
            )
        started = perf_counter()
        try:
            results = self._website_evidence.collect(str(candidate.source_url))
        except WebsiteEvidenceError as error:
            if record:
                record(
                    RunEventType.TOOL_FAILED,
                    {
                        "invocation_id": invocation_id,
                        "tool_name": "website_evidence",
                        "code": "website_evidence_failed",
                        "message": str(error),
                        "latency_ms": round((perf_counter() - started) * 1000),
                    },
                )
            return []
        if record:
            record(
                RunEventType.TOOL_COMPLETED,
                {
                    "invocation_id": invocation_id,
                    "tool_name": "website_evidence",
                    "result": {
                        "page_count": len(results),
                        "source_urls": [str(result.url) for result in results],
                    },
                    "latency_ms": round((perf_counter() - started) * 1000),
                },
            )
        return results

    def _generate_qualification(
        self,
        candidate: DiscoveryCandidate,
        evidence: list[SearchResult],
        model_id: str,
        instruction_appendix: str,
        record: Callable[[RunEventType, JsonObject], None] | None = None,
        approved_memory: JsonObject | None = None,
    ) -> tuple[GeneratedLeadQualification, QualificationGeneration]:
        context = candidate.discovery_context
        if context is None:
            raise DiscoveryAutopilotError("A discovery candidate is missing its opportunity lens.")

        def evidence_json(max_items: int, max_chars: int) -> str:
            compact_evidence = [
                {
                    "title": _bounded_text(item.title, 160),
                    "url": str(item.url),
                    "source_type": item.source_type,
                    "description": _bounded_text_with_tail(item.description, max_chars),
                }
                for item in evidence[:max_items]
            ]
            return json.dumps(compact_evidence, ensure_ascii=False)

        memory_json = _bounded_text_with_tail(
            json.dumps(approved_memory or {"semantic": [], "procedural": []}, ensure_ascii=False),
            _QUALIFICATION_MEMORY_MAX_CHARS,
        )
        instruction_text = _bounded_text_with_tail(
            instruction_appendix or "None.", _QUALIFICATION_INSTRUCTION_MAX_CHARS
        )

        def qualification_prompt(*, compact: bool) -> str:
            max_items = (
                _QUALIFICATION_RETRY_MAX_EVIDENCE if compact else _QUALIFICATION_MAX_EVIDENCE
            )
            max_chars = (
                _QUALIFICATION_RETRY_EVIDENCE_MAX_CHARS
                if compact
                else _QUALIFICATION_EVIDENCE_MAX_CHARS
            )
            evidence_block = evidence_json(max_items, max_chars)
            allowed_urls = json.dumps(
                [str(item.url) for item in evidence[:max_items]], ensure_ascii=False
            )
            return (
                "Verify whether this result represents one actual business. Return exactly "
                "one JSON object and no markdown or commentary. Use actual_company only when the "
                "supplied evidence supports one organization or its official business site. "
                "Reject listings, directories, rankings, comparison pages, marketplaces, lead "
                "databases, software/product pages, agencies, and ambiguous results. "
                "entity_reason must cite the supplied evidence. For actual_company, use a concise "
                "public brand/legal name, not a page headline; include summary, one or two "
                "company_signals, workflow_fit, a qualitative value_hypothesis, known_unknowns, "
                "and an integer confidence. Every claim source_urls list must contain only a URL "
                "from the supplied evidence. For non-company results, set rejection_reason and set "
                "the qualification fields to null or []. Keep statements under 220 characters, "
                "summary and value_hypothesis under 300 characters, and known_unknowns to at most "
                "three short items. Never invent metrics, contacts, customer relationships, or "
                "current tooling; keep human review in the workflow. Do not add any keys beyond "
                "this shape: "
                '{"entity_type":"actual_company","company_name":"...",'
                '"entity_reason":{"statement":"...","source_urls":["..."]},'
                '"rejection_reason":null,"summary":"...","company_signals":[{"statement":"...",'
                '"source_urls":["..."]}],"workflow_fit":{"statement":"...",'
                '"source_urls":["..."]},"value_hypothesis":"...",'
                '"known_unknowns":[],"confidence":0}.\n'
                f"Company: {_bounded_text(candidate.company_name, 180)}\n"
                f"Opportunity lens: {context.model_dump(mode='json')}\n"
                f"Public evidence (use only these URLs): {evidence_block}\n"
                f"Allowed source URLs: {allowed_urls}\n"
                + (
                    "Approved agent memory is context only, not public evidence: "
                    f"{memory_json}\nClient instruction appendix: {instruction_text}"
                    if not compact
                    else "Do not use memory or instructions as evidence."
                )
            )

        prompt = qualification_prompt(compact=False)
        compact_prompt = qualification_prompt(compact=True)
        errors: list[str] = []
        for attempt in range(1, self._validation_attempts + 1):
            try:
                if record:
                    record(
                        RunEventType.MODEL_STARTED,
                        {"operation": "lead_qualification", "attempt": attempt},
                    )
                response = self._provider.generate(
                    GenerationRequest(
                        system_prompt=(
                            "You are a cautious B2B qualification analyst. Never invent company "
                            "facts or customer relationships. Return JSON only."
                        ),
                        user_prompt=(
                            prompt
                            if attempt == 1
                            else f"{compact_prompt}\nPrevious response error: {errors[-1]}"
                        ),
                        operation="lead_qualification",
                        max_new_tokens=_QUALIFICATION_MAX_NEW_TOKENS,
                        agent_id="lead.discovery",
                    )
                )
                if record:
                    record(
                        RunEventType.MODEL_COMPLETED,
                        {
                            "operation": "lead_qualification",
                            "attempt": attempt,
                            "latency_ms": response.latency_ms,
                            "input_tokens": response.input_tokens,
                            "output_tokens": response.output_tokens,
                        },
                    )
                parsed = parse_and_validate_object(
                    response.content, GeneratedLeadQualification.model_json_schema()
                )
                return GeneratedLeadQualification.model_validate(parsed), QualificationGeneration(
                    model_id=model_id,
                    latency_ms=response.latency_ms,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    validation_attempts=attempt,
                )
            except ModelOutputTruncatedError as error:
                if record:
                    record(
                        RunEventType.MODEL_FAILED,
                        {
                            "operation": "lead_qualification",
                            "attempt": attempt,
                            "code": "model_output_truncated",
                            "message": str(error),
                            **error.metrics(),
                        },
                    )
                errors.append(
                    "The previous response reached the output limit. Return a concise JSON "
                    "object with complete required fields and no extra commentary."
                )
            except (ContractValidationError, ValueError) as error:
                message = str(error)
                errors.append(message)
                if record:
                    record(
                        RunEventType.MODEL_FAILED,
                        {
                            "operation": "lead_qualification",
                            "attempt": attempt,
                            "code": "model_output_invalid",
                            "message": message,
                        },
                    )
            except BudgetExceeded as error:
                if record:
                    record(
                        RunEventType.MODEL_FAILED,
                        {
                            "operation": "lead_qualification",
                            "attempt": attempt,
                            "code": "budget_exceeded",
                            "message": str(error)[:400],
                        },
                    )
                raise
            except ModelProviderError as error:
                if record:
                    record(
                        RunEventType.MODEL_FAILED,
                        {
                            "operation": "lead_qualification",
                            "attempt": attempt,
                            "code": "model_execution_failed",
                            "message": str(error),
                        },
                    )
                raise DiscoveryAutopilotError(
                    "The configured local model is unavailable for lead qualification."
                ) from error
        raise DiscoveryAutopilotError(
            "The local model could not produce a valid lead qualification brief."
        )

    def _plan(
        self,
        deployment_id: str,
        documents: list[KnowledgeDocument],
        existing_prospects: list[Prospect],
        record: Callable[[RunEventType, JsonObject], None] | None = None,
        approved_memory: JsonObject | None = None,
    ) -> TargetPlan:
        del deployment_id
        # Keep the planning prompt inside the local model's context window. The
        # full knowledge document is useful during research, but discovery only
        # needs a compact offer summary and source identifiers. When the prompt
        # was truncated, the model saw a lane item without the enclosing plan
        # contract and repeatedly returned a lane object.
        evidence, tracked, memory = self._planning_context(
            documents,
            existing_prospects,
            approved_memory,
            compact=False,
        )
        prompt = self._planning_prompt(evidence, tracked, memory, compact=False)
        compact_evidence, compact_tracked, compact_memory = self._planning_context(
            documents,
            existing_prospects,
            approved_memory,
            compact=True,
        )
        compact_prompt = self._planning_prompt(
            compact_evidence,
            compact_tracked,
            compact_memory,
            compact=True,
        )
        errors: list[str] = []
        for attempt in range(1, self._validation_attempts + 1):
            try:
                if record:
                    record(
                        RunEventType.MODEL_STARTED,
                        {"operation": "discovery_autopilot", "attempt": attempt},
                    )
                response = self._provider.generate(
                    GenerationRequest(
                        system_prompt=(
                            "You produce cautious, evidence-bound B2B targeting plans. "
                            "Return JSON only."
                        ),
                        user_prompt=(
                            compact_prompt
                            if attempt > 1 and errors and errors[-1].startswith("TRUNCATED:")
                            else prompt
                            if not errors
                            else f"{prompt}\nCorrection: {errors[-1]}"
                        ),
                        operation="discovery_autopilot",
                        max_new_tokens=(
                            _PLANNER_RETRY_MAX_NEW_TOKENS
                            if attempt > 1
                            else _PLANNER_MAX_NEW_TOKENS
                        ),
                        agent_id="lead.discovery",
                    )
                )
                if record:
                    record(
                        RunEventType.MODEL_COMPLETED,
                        {
                            "operation": "discovery_autopilot",
                            "attempt": attempt,
                            "latency_ms": response.latency_ms,
                            "input_tokens": response.input_tokens,
                            "output_tokens": response.output_tokens,
                        },
                    )
                return TargetPlan.model_validate(
                    parse_and_validate_object(response.content, TargetPlan.model_json_schema())
                )
            except (ContractValidationError, ValueError) as error:
                message = str(error)
                errors.append(message)
                if record:
                    record(
                        RunEventType.MODEL_FAILED,
                        {
                            "operation": "discovery_autopilot",
                            "attempt": attempt,
                            "code": "model_output_invalid",
                            "message": message,
                        },
                    )
            except ModelOutputTruncatedError as error:
                if record:
                    record(
                        RunEventType.MODEL_FAILED,
                        {
                            "operation": "discovery_autopilot",
                            "attempt": attempt,
                            "code": "model_output_truncated",
                            "message": str(error),
                            **error.metrics(),
                        },
                    )
                errors.append(
                    "TRUNCATED: Return only the compact JSON plan object; do not add commentary."
                )
                if attempt > 1 or attempt == self._validation_attempts:
                    raise DiscoveryAutopilotError(
                        "The local model could not finish the compact discovery plan within its "
                        f"{_PLANNER_RETRY_MAX_NEW_TOKENS}-token retry budget."
                    ) from error
            except BudgetExceeded as error:
                if record:
                    record(
                        RunEventType.MODEL_FAILED,
                        {
                            "operation": "discovery_autopilot",
                            "attempt": attempt,
                            "code": "budget_exceeded",
                            "message": str(error)[:400],
                        },
                    )
                raise
            except ModelProviderError as error:
                if record:
                    record(
                        RunEventType.MODEL_FAILED,
                        {
                            "operation": "discovery_autopilot",
                            "attempt": attempt,
                            "code": "model_execution_failed",
                            "message": str(error),
                        },
                    )
                raise DiscoveryAutopilotError(
                    "The configured local model is unavailable."
                ) from error
        raise DiscoveryAutopilotError("The local model could not produce a valid target plan.")

    def _planning_context(
        self,
        documents: list[KnowledgeDocument],
        existing_prospects: list[Prospect],
        approved_memory: JsonObject | None,
        *,
        compact: bool,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], JsonObject]:
        document_limit = (
            _PLANNER_RETRY_DOCUMENT_MAX_CHARS if compact else _PLANNER_DOCUMENT_MAX_CHARS
        )
        document_count = 2 if compact else 4
        evidence = [
            {
                "document_id": str(doc.document_id),
                "title": _bounded_text(doc.title, 120),
                "content": _compact_discovery_context(doc.content, max_chars=document_limit),
            }
            for doc in documents[:document_count]
        ]
        prospect_limit = _PLANNER_RETRY_MAX_TRACKED if compact else _PLANNER_MAX_TRACKED
        tracked: list[dict[str, object]] = []
        for prospect in existing_prospects[:prospect_limit]:
            domains = sorted(
                {
                    self._source_host(source.source_url)
                    for source in prospect.sources
                    if self._source_host(source.source_url)
                }
            )
            item: dict[str, object] = {"company": _bounded_text(prospect.company_name, 120)}
            if domains:
                item["domains"] = domains[:3]
            if (
                not compact
                and prospect.discovery_context
                and prospect.discovery_context.search_query
            ):
                item["previous_query"] = _bounded_text(prospect.discovery_context.search_query, 180)
            if prospect.archive_reason and not compact:
                item["archive_reason"] = _bounded_text(prospect.archive_reason, 120)
            tracked.append(item)

        source_memory = approved_memory or {"semantic": [], "procedural": []}
        memory_limit = _PLANNER_RETRY_MAX_MEMORY_ITEMS if compact else _PLANNER_MAX_MEMORY_ITEMS
        memory: JsonObject = {
            "semantic": [
                {
                    "subject": _bounded_text(item.get("subject"), 80),
                    "predicate": _bounded_text(item.get("predicate"), 80),
                    "value": _bounded_text(
                        json.dumps(item.get("value"), ensure_ascii=False)
                        if isinstance(item.get("value"), (dict, list))
                        else item.get("value"),
                        1_200 if item.get("predicate") == "ayni.company.market_profile" else 180,
                    ),
                }
                for item in (source_memory.get("semantic") or [])[:memory_limit]
            ],
            "procedural": [
                {
                    "title": _bounded_text(item.get("title"), 100),
                    "instruction": _bounded_text(item.get("instruction"), 180),
                }
                for item in (source_memory.get("procedural") or [])[:memory_limit]
            ],
        }
        return evidence, tracked, memory

    @staticmethod
    def _planning_prompt(
        evidence: list[dict[str, object]],
        tracked: list[dict[str, object]],
        memory: JsonObject,
        *,
        compact: bool,
    ) -> str:
        field_limits = (
            "Use short phrases: profile and proposition <=240 characters; each exclusion <=80; "
            "each sector <=80; each workflow and hypothesis <=180; each query <=140."
            if compact
            else "Use concise phrases: profile and proposition <=360 characters; each exclusion "
            "<=100; each sector <=100; each workflow and hypothesis <=240; each query <=180."
        )
        return (
            "Return exactly one JSON object for a new B2B discovery cycle. Use only the "
            "approved client context below. The value proposition must name a capability in that "
            "context. Exclude direct competitors, agencies, jobs, directories, and news. Create "
            "exactly three distinct-sector lanes: one validated lane and two exploratory lanes. "
            "Each query must describe a likely customer operation or pain, not generic AI terms. "
            "Use a geography only when the approved context explicitly identifies it as a target "
            "market or service region. Never infer target geography from a company headquarters or "
            "current location. If no target region is stated, keep search queries "
            "geography-neutral. "
            "Keep lane value hypotheses qualitative and testable; never invent savings percentages "
            "or volumes. Do not repeat tracked companies, domains, or prior queries. "
            f"{field_limits} Return JSON only, with no markdown or explanation.\n\n"
            "Required shape (do not return a lane object by itself): "
            '{"ideal_customer_profile":"...","value_proposition":"...",'
            '"exclusion_signals":["..."],"search_lanes":[{"sector":"...",'
            '"workflow":"...","value_hypothesis":"...","search_query":"...",'
            '"mode":"validated"},{"sector":"...","workflow":"...",'
            '"value_hypothesis":"...","search_query":"...","mode":"exploratory"},'
            '{"sector":"...","workflow":"...","value_hypothesis":"...",'
            '"search_query":"...","mode":"exploratory"}],"source_document_ids":["..."]}\n\n'
            f"Approved client context: {json.dumps(evidence, ensure_ascii=False)}\n"
            f"Approved agent memory (context only): {json.dumps(memory, ensure_ascii=False)}\n"
            f"Already tracked: {json.dumps(tracked, ensure_ascii=False)}\n"
            f"Cycle date: {utc_now().date().isoformat()}"
        )

    def _search_results(
        self,
        deployment_id: str,
        query: str,
        max_results: int,
        record: Callable[[RunEventType, JsonObject], None] | None,
        *,
        offset: int = 0,
    ) -> list[SearchResult]:
        invocation_id = str(uuid4())
        if record:
            record(
                RunEventType.TOOL_REQUESTED,
                {
                    "invocation_id": invocation_id,
                    "tool_name": "web_search",
                    "arguments": {"query": query, "max_results": max_results, "offset": offset},
                },
            )
        started = perf_counter()
        provider = self._search.resolve(deployment_id)
        try:
            results = provider.search(query, max_results=max_results, offset=offset)
        except Exception as error:
            if record:
                failure_payload: JsonObject = {
                    "invocation_id": invocation_id,
                    "tool_name": "web_search",
                    "code": "search_failed",
                    "message": str(error),
                    "latency_ms": round((perf_counter() - started) * 1000),
                }
                diagnostics = getattr(provider, "last_diagnostics", None)
                if diagnostics:
                    failure_payload["providers"] = diagnostics
                record(
                    RunEventType.TOOL_FAILED,
                    failure_payload,
                )
            raise
        if record:
            result_payload: JsonObject = {
                "result_count": len(results),
            }
            diagnostics = getattr(provider, "last_diagnostics", None)
            if diagnostics:
                result_payload["providers"] = diagnostics
            record(
                RunEventType.TOOL_COMPLETED,
                {
                    "invocation_id": invocation_id,
                    "tool_name": "web_search",
                    "result": result_payload,
                    "latency_ms": round((perf_counter() - started) * 1000),
                },
            )
        return results
