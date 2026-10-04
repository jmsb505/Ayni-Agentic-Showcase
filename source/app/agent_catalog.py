"""The Ayni-owned application agent roster.

The CRM migration keeps the durable runtime, approvals, and inspection model, but
the capabilities themselves belong to Ayni.  This module is the single contract
source for that roster.  It deliberately contains no provider or deployment
credentials: model selection remains a deployment-overlay concern.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.assistant_contracts import AssistantInput, AssistantOutput
from app.domain import (
    AgentCreate,
    AgentVersionCreate,
    JsonObject,
    PolicyOutcome,
    object_schema,
)
from app.persistence import RecordConflictError, RecordNotFoundError

if TYPE_CHECKING:
    from app.persistence import Repository
    from app.runtime import RegistryService


_PARITY = "retained_rebuilt"
_VERSION = "ayni-v1"


class AgentDecisionPoint(BaseModel):
    decision_id: str = Field(min_length=1, max_length=100, pattern=r"^[a-z][a-z0-9_]*$")
    method: Literal["judge", "deterministic", "human"]
    decision: str = Field(min_length=1, max_length=500)
    reason: str = Field(min_length=1, max_length=500)
    inputs: list[str] = Field(min_length=1, max_length=12)
    rule: str = Field(min_length=1, max_length=200)
    impact: str = Field(min_length=1, max_length=200)
    budget: str = Field(min_length=1, max_length=300)
    fallback: str = Field(min_length=1, max_length=300)
    trace: str = Field(min_length=1, max_length=300)
    measure: str = Field(min_length=1, max_length=300)

    model_config = ConfigDict(extra="forbid", frozen=True)


@dataclass(frozen=True)
class ApplicationAgentSpec:
    """Immutable application contract for one Ayni workflow capability."""

    agent_id: str
    name: str
    description: str
    workflow_stage: str
    source_capability: str
    harness: str
    instructions: str
    input_schema: JsonObject
    output_schema: JsonObject
    decision_scope: Literal["data_only", "workflow"]
    decision_points: tuple[AgentDecisionPoint, ...]
    allowed_tools: tuple[str, ...] = ()
    allowed_actions: tuple[str, ...] = ()
    action_policy: PolicyOutcome = PolicyOutcome.ALLOW
    version: str = _VERSION

    def __post_init__(self) -> None:
        if self.decision_scope == "data_only" and self.decision_points:
            raise ValueError("Data-only capabilities cannot declare workflow decision points")
        if self.decision_scope == "workflow" and not self.decision_points:
            raise ValueError("Workflow capabilities must declare their decision points")
        identifiers = [point.decision_id for point in self.decision_points]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Decision IDs must be unique within a capability")

    @property
    def disposition(self) -> str:
        return _PARITY

    def _metadata(self) -> JsonObject:
        return {
            "application_owned": True,
            "parity": _PARITY,
            "capability_id": self.agent_id,
            "workflow_stage": self.workflow_stage,
            "source_capability": self.source_capability,
            "execution_boundary": "ayni",
            "model_binding": "deployment_overlay",
            "disposition": _PARITY,
            "decision_scope": self.decision_scope,
            "decision_points": [point.model_dump(mode="json") for point in self.decision_points],
        }

    def version_request(self) -> AgentVersionCreate:
        return AgentVersionCreate(
            version=self.version,
            instructions=self.instructions,
            harness=self.harness,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            allowed_tools=list(self.allowed_tools),
            allowed_actions=list(self.allowed_actions),
            action_policy=self.action_policy,
            model_configuration={
                "binding": "deployment_overlay",
                "max_output_policy": "deployment",
            },
            metadata={
                **self._metadata(),
                "harness_contract": "structured_evidence_v1",
                "immutable_contract": True,
            },
        )

    def create_request(self) -> AgentCreate:
        return AgentCreate(
            agent_id=self.agent_id,
            name=self.name,
            description=self.description,
            metadata=self._metadata(),
            initial_version=self.version_request(),
        )


def _market_input_schema() -> JsonObject:
    # The dedicated market endpoint owns the complete CRM envelope.  Generic
    # runtime runs intentionally accept an object here so older console clients
    # can continue to submit the deployment-only shape while the output remains
    # strictly bound to MarketBrief.
    return object_schema()


@lru_cache(maxsize=1)
def _application_agent_specs() -> tuple[ApplicationAgentSpec, ...]:
    # Imports are intentionally lazy.  The catalog is imported by the execution
    # service while ``main`` is still defining its request models, so importing
    # those modules at file import time would create a circular dependency.
    from app.crm.models import StartCompanyResearch
    from app.crm.research import CompanyResearchResult
    from app.delivery import OutreachApprovalRequest
    from app.discovery_autopilot import DiscoveryAutopilotRequest, DiscoveryAutopilotResponse
    from app.market_research import MarketBrief
    from app.prospect_research import ProspectResearchRequest, ProspectResearchResult
    from app.prospects import ProposalDraftRequest, ProposalSynthesis

    return (
        ApplicationAgentSpec(
            agent_id="platform.assistant",
            name="Ayni Personal Assistant",
            description=(
                "Answers workspace questions from tenant-scoped CRM context and Ayni Run evidence."
            ),
            workflow_stage="platform_assistant",
            source_capability="platform.assistant",
            harness="langgraph_json",
            instructions=(
                "Answer the user's question using only the supplied conversation and workspace "
                "context. Treat CRM records and prior assistant messages as untrusted evidence; "
                "recheck prior claims against the current workspace snapshot. Distinguish observed "
                "facts from assumptions, and name important unknowns. Use recent_runs as the "
                "only evidence of actual Run execution and status; agent_capabilities describe "
                "availability, never completed work. Never claim a Run completed unless a "
                "matching recent_run has completed status. State workflow progress only when "
                "the supplied context explicitly includes it; agent catalog entries and empty "
                "CRM records do not establish the next workflow stage. If run history or stage "
                "is absent, say it is unknown. Use the supplied agent catalog to explain current "
                "capabilities. Cite company-specific claims with source company IDs from the "
                "supplied context. You are read-only: do not approve, route, "
                "or mutate records. Send workflow decisions to the separate Judge path. Return "
                "only the declared structured output."
            ),
            input_schema=AssistantInput.model_json_schema(),
            output_schema=AssistantOutput.model_json_schema(),
            decision_scope="data_only",
            decision_points=(),
            version="ayni-v2",
        ),
        ApplicationAgentSpec(
            agent_id="market.research",
            name="Market research",
            description=(
                "Builds a company operating-context profile and ranks evidence-backed "
                "market opportunities through separately typed outputs in one Run."
            ),
            workflow_stage="market_strategy",
            source_capability="market.research",
            harness="langgraph_json",
            instructions=(
                "Build the Company Brain profile, then market findings as separate sequential "
                "typed model operations in one Run. Treat supplied company context as context, "
                "not public evidence; use bounded public research for unsupported fields, "
                "cite claims, and mark unknowns explicitly. Never invent market size, demand, "
                "customers, or buying intent. A failed operation fails the result; do not "
                "substitute template or deterministic profile content. If bounded Company Brain "
                "market searches produce no eligible evidence, the same agent may author at most "
                "two 2-18 word retrieval queries per lane from founder context and rejected result "
                "titles/hosts/reasons, preserving a specific term from that lane; allow at most "
                "one validation-only repair per query and search only each final valid query. "
                "Reuse the same "
                "source gate; user-uploaded scholarly networking hosts are not primary evidence, "
                "and title-format variants of one source are not independent citations. The query "
                "and its search "
                "snippets are not evidence. Record screened titles/hosts and rejection reasons, "
                "never excerpts, in the Run trace. Provider unavailability skips refinement, and "
                "no eligible source means the finding stays unknown. Preserve declared markets; "
                "the explicit phrase 'small and mid-sized businesses' may be abbreviated as SMB "
                "in a retrieval query, but declared geography must not be inferred or omitted."
            ),
            input_schema=_market_input_schema(),
            output_schema=MarketBrief.model_json_schema(),
            decision_scope="workflow",
            decision_points=(
                AgentDecisionPoint(
                    decision_id="profile_claim_grounding",
                    method="deterministic",
                    decision="Whether generated profile claims satisfy citation and source rules.",
                    reason="Citation IDs, evidence scope, and numeric claims have exact checks.",
                    inputs=["profile claims", "source references", "authorized evidence"],
                    rule="market-profile-evidence.v1",
                    impact="Medium: invalid claims cannot become trusted profile facts.",
                    budget="No additional model call.",
                    fallback="Reject invalid claims and leave unsupported fields unknown.",
                    trace="Record validation outcome and safe source references in the Run.",
                    measure="Invalid-claim rate and operator corrections.",
                ),
                AgentDecisionPoint(
                    decision_id="profile_readiness",
                    method="judge",
                    decision="Whether the profile is ready, needs revision, or lacks evidence.",
                    reason="Readiness depends on contextual evidence across profile areas.",
                    inputs=["bounded profile state", "tenant-scoped evidence", "question version"],
                    rule="company-profile-review.v3 and its per-decision thresholds",
                    impact="High when an owner enables autonomous profile approval.",
                    budget="One batched Judge request and at most one bounded correction.",
                    fallback="Keep the profile in human review on missing or uncertain results.",
                    trace=(
                        "Record typed answers, evidence references, policy version, and "
                        "correction status."
                    ),
                    measure="Held-out quality, calibration, cost, latency, and review effort.",
                ),
                AgentDecisionPoint(
                    decision_id="profile_approval",
                    method="human",
                    decision=(
                        "Whether the reviewed profile may become approved organization memory."
                    ),
                    reason="The profile becomes reusable company context for later workflows.",
                    inputs=["current profile Run", "source references", "organization revision"],
                    rule=(
                        "Workspace-owner approval; validated Auto policy is a separately "
                        "enabled path."
                    ),
                    impact="High.",
                    budget="One review per profile revision.",
                    fallback="Leave the current approved profile unchanged.",
                    trace="Record approver, Run, memory reference, and profile revision.",
                    measure="Approval corrections and downstream profile changes.",
                ),
            ),
            # V46 allows workflow-case retrieval without forcing SMB wording.
            version="ayni-v46",
        ),
        ApplicationAgentSpec(
            agent_id="lead.discovery",
            name="Automated lead discovery",
            description="Plans, searches, screens, and qualifies bounded lead candidates.",
            workflow_stage="lead_discovery",
            source_capability="lead.discovery",
            harness="langgraph_json",
            instructions=(
                "Create cautious, evidence-bound targeting plans and qualification briefs. "
                "Use only supplied evidence, reject ambiguous entities, and keep human review "
                "visible before a prospect advances."
            ),
            input_schema=DiscoveryAutopilotRequest.model_json_schema(),
            output_schema=DiscoveryAutopilotResponse.model_json_schema(),
            decision_scope="workflow",
            decision_points=(
                AgentDecisionPoint(
                    decision_id="candidate_hard_eligibility",
                    method="deterministic",
                    decision="Whether a candidate passes exact eligibility and duplicate checks.",
                    reason="Hard requirements and identity matching must be reproducible.",
                    inputs=[
                        "candidate identifiers",
                        "approved profile constraints",
                        "tenant records",
                    ],
                    rule="company-lead-eligibility.v1",
                    impact="Medium.",
                    budget="No additional model call.",
                    fallback="Reject or hold ambiguous candidates for review.",
                    trace="Record rule version and candidate disposition in Run evidence.",
                    measure="Duplicate rate and eligibility corrections.",
                ),
                AgentDecisionPoint(
                    decision_id="lead_fit_and_route",
                    method="judge",
                    decision=(
                        "Whether an eligible lead fits the approved target profile and may advance."
                    ),
                    reason="Fit and ambiguity depend on contextual evidence.",
                    inputs=["eligible candidates", "approved profile", "scoped evidence"],
                    rule="company-lead-routing.v2 and per-mode confidence thresholds",
                    impact="Medium: Auto can start downstream research and drafting.",
                    budget="One batched Judge request within the workflow budget.",
                    fallback="Keep candidates pending review or stop Auto routing.",
                    trace=(
                        "Record typed disposition, confidence, policy version, and candidate "
                        "references."
                    ),
                    measure="Held-out routing quality, calibration, cost, latency, and overrides.",
                ),
                AgentDecisionPoint(
                    decision_id="lead_review",
                    method="human",
                    decision="Whether a queued candidate should enter deep research.",
                    reason="The standard workflow keeps lead advancement under owner control.",
                    inputs=["candidate evidence", "qualification result", "approved profile"],
                    rule="Owner approval or separately enabled autonomous routing policy.",
                    impact="Medium.",
                    budget="One decision per candidate.",
                    fallback="Keep the candidate pending review.",
                    trace="Link the operator decision to the candidate and discovery Run.",
                    measure="Approval rate, reversals, and downstream usefulness.",
                ),
            ),
        ),
        ApplicationAgentSpec(
            agent_id="lead.research",
            name="Lead research",
            description="Builds cited prospect profiles, fit assessments, and bounded unknowns.",
            workflow_stage="lead_research",
            source_capability="lead.research",
            harness="langgraph_json",
            instructions=(
                "Research one retained prospect using only approved evidence. Cite every "
                "claim, separate facts from hypotheses, and expose delivery risks and unknowns."
            ),
            input_schema=ProspectResearchRequest.model_json_schema(),
            output_schema=ProspectResearchResult.model_json_schema(),
            decision_scope="workflow",
            decision_points=(
                AgentDecisionPoint(
                    decision_id="prospect_claim_validation",
                    method="deterministic",
                    decision="Whether research claims use valid, in-scope source references.",
                    reason="Source IDs and numeric-claim requirements are exact checks.",
                    inputs=["research claims", "retained evidence", "source IDs"],
                    rule="prospect-research-evidence.v1",
                    impact="Medium.",
                    budget="No additional model call.",
                    fallback="Reject the invalid result and retain the Prospect for review.",
                    trace="Record validation errors and safe evidence references in the Run.",
                    measure="Unsupported-claim rate and operator corrections.",
                ),
                AgentDecisionPoint(
                    decision_id="lead_research_quality",
                    method="judge",
                    decision=(
                        "Whether cited research is evidence-grounded, workflow-specific, and "
                        "appropriately uncertain."
                    ),
                    reason=(
                        "Semantic evidence support and fit need a typed Judge assessment; exact "
                        "source membership remains deterministic."
                    ),
                    inputs=[
                        "approved company context",
                        "lead qualification",
                        "cited assessment",
                        "retained source excerpts",
                        "approved memory context",
                    ],
                    rule="lead-research-quality.v1; lead-artifact-quality.v1",
                    impact="Medium; does not approve a lead or authorize a proposal.",
                    budget=(
                        "One batched Judge call; Assist/Autonomous may make one agent correction "
                        "and one fresh recheck."
                    ),
                    fallback=(
                        "Observe records an unavailable result; Assist/Autonomous save no "
                        "assessment if Judge review or recheck fails."
                    ),
                    trace=(
                        "Record the typed request, result, input digest, policy decision, "
                        "and usage in the Run."
                    ),
                    measure=(
                        "Judge/operator agreement, unsupported-claim rate, corrections, "
                        "latency, and cost."
                    ),
                ),
                AgentDecisionPoint(
                    decision_id="prospect_research_acceptance",
                    method="human",
                    decision="Whether cited research is sufficient to inform a proposal.",
                    reason="Evidence sufficiency and business relevance require contextual review.",
                    inputs=["validated research", "source references", "approved company profile"],
                    rule=(
                        "Human review remains required after the optional Judge quality assessment."
                    ),
                    impact="Medium.",
                    budget="One review per research Run.",
                    fallback="Do not start proposal drafting until reviewed.",
                    trace="Record reviewer outcome and link it to the research Run.",
                    measure="Review corrections and proposal-stage rework.",
                ),
            ),
        ),
        ApplicationAgentSpec(
            agent_id="lead.proposal",
            name="Lead proposal",
            description="Creates a lead-specific, evidence-cited proposal for human review.",
            workflow_stage="proposal",
            source_capability="lead.proposal",
            harness="langgraph_json",
            instructions=(
                "Write a concise B2B proposal from retained qualification and research. "
                "Use only cited evidence, keep unknowns explicit, and never send outreach."
            ),
            input_schema=ProposalDraftRequest.model_json_schema(),
            output_schema=ProposalSynthesis.model_json_schema(),
            decision_scope="workflow",
            decision_points=(
                AgentDecisionPoint(
                    decision_id="proposal_claim_validation",
                    method="deterministic",
                    decision=(
                        "Whether proposal claims, citations, and delivery constraints are valid."
                    ),
                    reason=(
                        "Citation membership and exact payload constraints are machine-checkable."
                    ),
                    inputs=["proposal draft", "retained research", "delivery constraints"],
                    rule="proposal-evidence-and-delivery.v1",
                    impact="Medium.",
                    budget="No additional model call.",
                    fallback="Reject invalid content and keep delivery disabled.",
                    trace="Record validation outcome and source references in the Run.",
                    measure="Unsupported-claim rate and proposal rework.",
                ),
                AgentDecisionPoint(
                    decision_id="lead_proposal_quality",
                    method="judge",
                    decision=(
                        "Whether proposal claims and outreach draft align with retained evidence, "
                        "lead-specific workflow signals, and uncertainty boundaries."
                    ),
                    reason=(
                        "Semantic fit and communication safety need a typed Judge assessment; "
                        "citation membership and delivery rules remain deterministic."
                    ),
                    inputs=[
                        "approved company context",
                        "lead qualification and research",
                        "proposal synthesis",
                        "retained source excerpts",
                        "approved memory context",
                    ],
                    rule="lead-proposal-quality.v1; lead-artifact-quality.v1",
                    impact="High before external communication; Judge cannot approve or send.",
                    budget=(
                        "One batched Judge call; Assist/Autonomous may make one agent correction "
                        "and one fresh recheck."
                    ),
                    fallback=(
                        "Observe records an unavailable result; Assist/Autonomous save no draft "
                        "if Judge review or recheck fails."
                    ),
                    trace=(
                        "Record the typed request, result, input digest, policy decision, "
                        "and usage in the Run."
                    ),
                    measure=(
                        "Judge/operator agreement, evidence alignment, draft edits, latency, "
                        "and cost."
                    ),
                ),
                AgentDecisionPoint(
                    decision_id="proposal_review",
                    method="human",
                    decision=(
                        "Whether the proposal and exact outreach draft are suitable to present."
                    ),
                    reason="Business fit and client voice require contextual review.",
                    inputs=["proposal", "approved company profile", "research evidence"],
                    rule="Human review; delivery remains a separate approved Action.",
                    impact="High before external communication.",
                    budget="One review per proposal version.",
                    fallback="Keep the draft review-required.",
                    trace="Record review status and link it to the proposal and Run.",
                    measure="Operator edits, rejection rate, and downstream response quality.",
                ),
            ),
        ),
        ApplicationAgentSpec(
            agent_id="crm.company-description-research",
            name="Company description research",
            description=(
                "Drafts an evidence-backed CRM Company description as a reviewable proposal."
            ),
            workflow_stage="crm_company_research",
            source_capability="crm.company-description-research",
            harness="langgraph_json",
            instructions=(
                "Draft a concise factual Company description from authorized CRM context and "
                "bounded public evidence. Cite every claim, do not mutate CRM data, and leave "
                "the proposal and approval boundary visible."
            ),
            input_schema=StartCompanyResearch.model_json_schema(),
            output_schema=CompanyResearchResult.model_json_schema(),
            decision_scope="workflow",
            decision_points=(
                AgentDecisionPoint(
                    decision_id="company_description_claim_validation",
                    method="deterministic",
                    decision="Whether proposed description claims are grounded in scoped sources.",
                    reason="Source membership and citation requirements are exact checks.",
                    inputs=["description claims", "CRM company context", "public evidence"],
                    rule="crm-company-description-evidence.v1",
                    impact="Medium.",
                    budget="No additional model call.",
                    fallback="Reject unsupported claims and keep the Company unchanged.",
                    trace="Record validation outcome and source references in the Run.",
                    measure="Unsupported-claim rate and operator corrections.",
                ),
                AgentDecisionPoint(
                    decision_id="company_description_approval",
                    method="human",
                    decision=(
                        "Whether the proposed description should be applied to the CRM Company."
                    ),
                    reason="The proposal changes an authoritative CRM business record.",
                    inputs=["proposal", "source references", "current CRM revision"],
                    rule="Owner review and the existing CRM approval path.",
                    impact="High.",
                    budget="One review per proposal revision.",
                    fallback="Leave the CRM Company unchanged.",
                    trace="Record approval and applied revision through the CRM action receipt.",
                    measure="Approval reversals and corrections after application.",
                ),
            ),
        ),
        ApplicationAgentSpec(
            agent_id="lead.outreach",
            name="Approved outreach",
            description=(
                "Delivers an exact, human-approved proposal message through a controlled action."
            ),
            workflow_stage="approved_delivery",
            source_capability="lead.outreach",
            harness="simple_json",
            instructions=(
                "This is an action-only delivery capability. Preserve the approved message "
                "exactly, request send_outreach only after human approval, and never rewrite "
                "or invent recipient content."
            ),
            input_schema=OutreachApprovalRequest.model_json_schema(),
            output_schema=object_schema(),
            allowed_actions=("send_outreach",),
            action_policy=PolicyOutcome.REQUIRE_APPROVAL,
            decision_scope="workflow",
            decision_points=(
                AgentDecisionPoint(
                    decision_id="outreach_exact_payload_approval",
                    method="human",
                    decision="Whether the exact recipient and message may be sent.",
                    reason="External communication needs explicit approval of the final payload.",
                    inputs=["recipient", "exact message", "approved proposal"],
                    rule="Action approval bound to the exact payload digest.",
                    impact="High.",
                    budget="One approval per exact payload.",
                    fallback="Do not send.",
                    trace="Record approver, payload digest, Action, and delivery receipt.",
                    measure="Delivery failures, corrections, and complaint rate.",
                ),
                AgentDecisionPoint(
                    decision_id="outreach_effect_reconciliation",
                    method="deterministic",
                    decision="Whether delivery succeeded, failed, or needs reconciliation.",
                    reason="A verified external receipt establishes effect status.",
                    inputs=["Action request", "provider receipt", "idempotency key"],
                    rule="action.receipt.v1 and reconcile-before-retry policy",
                    impact="High.",
                    budget="Reconcile before retrying an uncertain outcome.",
                    fallback="Mark uncertain and block duplicate delivery.",
                    trace="Retain the verified receipt or reconciliation reference.",
                    measure="Uncertain-outcome rate and duplicate-effect incidents.",
                ),
            ),
        ),
    )


def application_agent_specs() -> tuple[ApplicationAgentSpec, ...]:
    """Return the stable Ayni agent roster."""

    return _application_agent_specs()


def get_application_agent_spec(agent_id: str) -> ApplicationAgentSpec | None:
    return next((spec for spec in application_agent_specs() if spec.agent_id == agent_id), None)


def agent_capability_catalog() -> list[JsonObject]:
    """Return the public, schema-bearing parity catalog for the developer UI."""

    return [
        {
            "agent_id": spec.agent_id,
            "name": spec.name,
            "description": spec.description,
            "workflow_stage": spec.workflow_stage,
            "source_capability": spec.source_capability,
            "disposition": spec.disposition,
            "version": spec.version,
            "harness": spec.harness,
            "input_schema": spec.input_schema,
            "output_schema": spec.output_schema,
            "allowed_tools": list(spec.allowed_tools),
            "allowed_actions": list(spec.allowed_actions),
            "action_policy": spec.action_policy.value,
            "model_binding": "deployment_overlay",
            "execution_boundary": "ayni",
            "decision_scope": spec.decision_scope,
            "decision_points": list(spec.decision_points),
        }
        for spec in application_agent_specs()
    ]


def _matches_spec(version, spec: ApplicationAgentSpec) -> bool:
    metadata = version.metadata
    return (
        version.version == spec.version
        and version.harness == spec.harness
        and metadata.get("application_owned") is True
        and metadata.get("parity") == _PARITY
        and metadata.get("capability_id") == spec.agent_id
        and metadata.get("immutable_contract") is True
    )


def ensure_application_agents(
    repository: Repository,
    registry: RegistryService,
    *,
    agent_ids: Iterable[str] | None = None,
) -> dict[str, object]:
    """Register or promote every Ayni capability without changing old versions.

    Existing application-owned agents are retained, but a generic legacy current
    version is never silently reused.  Instead an immutable ``ayni-v1`` version
    is created and made current.  A same-name contract collision fails closed so
    a deployment cannot run with an ambiguous capability definition.
    """

    selected = set(agent_ids) if agent_ids is not None else None
    specs = [
        spec for spec in application_agent_specs() if selected is None or spec.agent_id in selected
    ]
    if selected is not None and len(specs) != len(selected):
        missing = sorted(selected - {spec.agent_id for spec in specs})
        raise ValueError(f"Unknown application agent(s): {', '.join(missing)}")

    ensured: dict[str, object] = {}
    for spec in specs:
        try:
            agent = repository.get_agent(spec.agent_id)
        except RecordNotFoundError:
            registered = registry.register(spec.create_request())
            if registered.initial_version is None:
                raise RecordConflictError(
                    f"Agent has no initial version: {spec.agent_id}"
                ) from None
            ensured[spec.agent_id] = registered.initial_version
            continue

        if agent.metadata.get("application_owned") is not True:
            raise RecordConflictError(
                f"Ayni capability '{spec.agent_id}' is owned by a non-application agent."
            )
        try:
            current = repository.resolve_version(spec.agent_id, None)
        except RecordNotFoundError:
            current = None
        if current is not None and _matches_spec(current, spec):
            ensured[spec.agent_id] = current
            continue

        try:
            version = repository.get_version(spec.agent_id, spec.version)
        except RecordNotFoundError:
            version = registry.create_version(
                spec.agent_id,
                spec.version_request(),
                make_current=True,
            )
        else:
            if not _matches_spec(version, spec):
                raise RecordConflictError(
                    f"Ayni capability contract collision: {spec.agent_id}@{spec.version}"
                )
            # The immutable version may exist but not be current after an
            # interrupted promotion.  The repository has no mutable agent API;
            # creating a duplicate is forbidden, so surface the drift clearly.
            if current is None or current.version_id != version.version_id:
                raise RecordConflictError(
                    f"Ayni capability current-version drift: {spec.agent_id}@{spec.version}"
                )
        ensured[spec.agent_id] = version

    return ensured
