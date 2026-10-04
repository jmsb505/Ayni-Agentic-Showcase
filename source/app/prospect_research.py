from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter
from typing import Literal
from urllib.parse import urlparse
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.config import Settings
from app.connectors.ports import JudgeChoiceQuestion, JudgeEvaluationRequest, JudgeEvaluationResult
from app.deployments import ClientDeployment, DeploymentCatalog
from app.domain import JsonObject, RunEventType
from app.execution_controls import BudgetExceeded, bounded_service
from app.judge_policy import (
    LEAD_ARTIFACT_QUALITY_POLICY_VERSION,
    JudgeEvaluator,
    JudgeMode,
    assess_artifact_quality_result,
    evaluate_judgment,
)
from app.knowledge_memory import (
    KnowledgeMemoryStore,
    KnowledgeSearchRequest,
    approved_agent_memory_context,
)
from app.model_provider import (
    GenerationRequest,
    ModelOutputTruncatedError,
    ModelProvider,
    ModelProviderError,
)
from app.persistence import RecordConflictError
from app.prospects import (
    BusinessProfile,
    CompetitorObservation,
    DeliveryFit,
    EvidenceCitation,
    Prospect,
    ProspectAssessment,
    ProspectAssessmentCreate,
    ProspectStatus,
    ProspectStore,
    ResearchGenerationEvidence,
)
from app.validation import ContractValidationError, parse_and_validate_object
from app.website_evidence import WebsiteEvidenceCollector, WebsiteEvidenceError

RESEARCH_AGENT_ID = "lead.research"
LEAD_RESEARCH_QUALITY_QUESTION_VERSION = "lead-research-quality.v1"


class ProspectResearchRequest(BaseModel):
    deployment_id: str = Field(min_length=1, max_length=80)
    judge_mode: Literal["off", "observe", "assist", "autonomous"] = "off"
    focus: str = Field(
        default="workflow fit, measurable value, delivery risk, and differentiation",
        min_length=8,
        max_length=500,
    )


class ClaimCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str = Field(min_length=1, max_length=1_000)
    citation_ids: list[str] = Field(min_length=1, max_length=8)


class ProceduralSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=300)
    instruction: str = Field(min_length=1, max_length=8_000)
    supporting_evidence: list[str] = Field(min_length=1, max_length=20)


class GeneratedAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    delivery_fit: DeliveryFit
    capacity_risk: str = Field(min_length=1, max_length=1_000)
    operational_signals: list[str] = Field(min_length=1, max_length=12)
    workflow_opportunities: list[str] = Field(min_length=1, max_length=8)
    competitors: list[CompetitorObservation] = Field(default_factory=list, max_length=12)
    value_hypothesis: str = Field(min_length=1, max_length=2_000)
    known_unknowns: list[str] = Field(default_factory=list, max_length=12)
    confidence: int = Field(ge=0, le=100)
    claim_citations: list[ClaimCitation] = Field(min_length=1, max_length=24)
    procedural_suggestions: list[ProceduralSuggestion] = Field(default_factory=list, max_length=4)
    business_profile: BusinessProfile | None = None

    @field_validator("operational_signals", "workflow_opportunities", "known_unknowns")
    @classmethod
    def unique_plain_statements(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("Research statements must not be empty.")
        if len({value.casefold() for value in normalized}) != len(normalized):
            raise ValueError("Research statements must not be duplicated.")
        return normalized

    @model_validator(mode="after")
    def every_citation_must_be_referenced(self) -> GeneratedAssessment:
        if not self.claim_citations:
            raise ValueError("Generated research must include cited claims.")
        return self

    @model_validator(mode="after")
    def procedural_suggestions_must_be_cited(self) -> GeneratedAssessment:
        if any(
            not reference.strip()
            for suggestion in self.procedural_suggestions
            for reference in suggestion.supporting_evidence
        ):
            raise ValueError("Procedural memory suggestions must include evidence references.")
        return self


class ProfileResearchPass(BaseModel):
    """Small model contract for evidence-backed company/contact extraction."""

    model_config = ConfigDict(extra="forbid")

    # Every completed research run exposes a compact profile. Unsupported fields are represented
    # explicitly as unknown/empty values instead of silently omitting the profile.
    business_profile: BusinessProfile
    claim_citations: list[ClaimCitation] = Field(default_factory=list, max_length=8)


class AssessmentResearchPass(BaseModel):
    """Small model contract for fit, value, competition, and uncertainty analysis."""

    model_config = ConfigDict(extra="forbid")

    delivery_fit: DeliveryFit
    capacity_risk: str = Field(min_length=1, max_length=500)
    operational_signals: list[str] = Field(min_length=1, max_length=6)
    workflow_opportunities: list[str] = Field(min_length=1, max_length=5)
    competitors: list[CompetitorObservation] = Field(default_factory=list, max_length=4)
    value_hypothesis: str = Field(min_length=1, max_length=800)
    known_unknowns: list[str] = Field(default_factory=list, max_length=6)
    confidence: int = Field(ge=0, le=100)
    claim_citations: list[ClaimCitation] = Field(min_length=1, max_length=8)
    procedural_suggestions: list[ProceduralSuggestion] = Field(default_factory=list, max_length=2)


class ProspectResearchResult(BaseModel):
    prospect: Prospect
    assessment: ProspectAssessment
    cited_claims: list[ClaimCitation]
    procedural_suggestions: list[ProceduralSuggestion] = Field(default_factory=list, max_length=4)


class ProspectResearchError(RuntimeError):
    pass


@dataclass(frozen=True)
class _ResearchContext:
    deployment: ClientDeployment
    evidence: list[EvidenceCitation]
    approved_memory: JsonObject


class ProspectResearchService:
    """Bounded, cited research for the M8 configured region workflow.

    This deliberately does not create platform Agents, Actions, or autonomous outreach. It is an
    application-layer experiment that uses one manifest-selected local model and persists its
    reviewable assessment alongside the supplied evidence.
    """

    def __init__(
        self,
        prospects: ProspectStore,
        knowledge: KnowledgeMemoryStore,
        deployments: DeploymentCatalog,
        model_provider: ModelProvider,
        settings: Settings,
        website_evidence: WebsiteEvidenceCollector | None = None,
        judge_evaluator: JudgeEvaluator | None = None,
    ) -> None:
        self._prospects = prospects
        self._knowledge = knowledge
        self._deployments = deployments
        self._model_provider = model_provider
        self._settings = settings
        self._website_evidence = website_evidence
        self._judge_evaluator = judge_evaluator

    @bounded_service
    def research(
        self,
        prospect_id: UUID,
        request: ProspectResearchRequest,
        record: Callable[[RunEventType, JsonObject], None] | None = None,
    ) -> ProspectResearchResult:
        prospect = self._prospects.get(prospect_id)
        if prospect.status in {ProspectStatus.PENDING_REVIEW, ProspectStatus.ARCHIVED}:
            raise RecordConflictError("Approve an active prospect before running research.")
        mode = JudgeMode(request.judge_mode)
        if mode is not JudgeMode.OFF and record is None:
            raise RecordConflictError(
                "Judge-enabled research must run through an agent Run so its evaluation is traced."
            )
        context = self._context_for(prospect, request, record)
        generated, generation = self._generate(prospect, request, context, record)
        if mode is not JudgeMode.OFF:
            generated, generation = self._review_generated(
                prospect, request, context, generated, generation, record
            )
        referenced_ids = self._validated_reference_ids(generated, context.evidence)
        assessment = self._prospects.create_assessment(
            prospect_id,
            ProspectAssessmentCreate(
                **generated.model_dump(exclude={"claim_citations", "procedural_suggestions"}),
                evidence=[
                    citation
                    for citation in context.evidence
                    if citation.citation_id in referenced_ids
                ],
                generation=generation,
            ),
        )
        return ProspectResearchResult(
            prospect=self._prospects.get(prospect_id),
            assessment=assessment,
            cited_claims=generated.claim_citations,
            procedural_suggestions=generated.procedural_suggestions,
        )

    def _review_generated(
        self,
        prospect: Prospect,
        request: ProspectResearchRequest,
        context: _ResearchContext,
        generated: GeneratedAssessment,
        generation: ResearchGenerationEvidence,
        record: Callable[[RunEventType, JsonObject], None],
    ) -> tuple[GeneratedAssessment, ResearchGenerationEvidence]:
        mode = JudgeMode(request.judge_mode)
        if mode is JudgeMode.OFF:
            return generated, generation

        review_id = str(uuid4())
        check_ids = ("evidence_support", "workflow_fit", "uncertainty_integrity")

        def evaluate(value: GeneratedAssessment, pass_number: int):
            judge_request = self._quality_judge_request(
                prospect,
                value,
                context,
                review_pass=pass_number,
            )
            return evaluate_judgment(
                mode,
                self._judge_evaluator,
                judge_request,
                decision_id=f"{review_id}:lead_research_quality:v1:pass_{pass_number}",
                deployment_id=request.deployment_id,
                record=record,
                requested_label="Judge reviews lead research",
                completed_label="Judge assessed lead research",
            )

        first_result = evaluate(generated, 1)
        if first_result is None:
            if self._judge_evaluator is None:
                record(
                    RunEventType.TOOL_FAILED,
                    {
                        "tool_name": "judge.evaluate",
                        "decision_id": f"{review_id}:lead_research_quality:v1:pass_1",
                        "failure_code": "judge_not_configured",
                        "display_label": "Lead research Judge is not configured",
                    },
                )
            if mode is JudgeMode.OBSERVE:
                return generated, generation
            raise ProspectResearchError(
                "The lead research Judge returned no result; no research assessment was saved."
            )

        decision, confidence, assessments, revision_fields = assess_artifact_quality_result(
            first_result,
            overall_question_id="research_review",
            check_question_ids=check_ids,
        )
        self._record_quality_decision(
            record,
            review_id=review_id,
            stage="lead.research",
            question_version=LEAD_RESEARCH_QUALITY_QUESTION_VERSION,
            pass_number=1,
            result=first_result,
            decision=decision,
            confidence=confidence,
            assessments=assessments,
            revision_fields=revision_fields,
        )
        if mode is JudgeMode.OBSERVE or decision == "ready":
            return generated, generation
        if decision != "needs_revision" or not revision_fields:
            raise ProspectResearchError(
                "The lead research Judge could not establish a reviewable result; "
                "no research assessment was saved."
            )

        corrected, correction_generation = self._generate(
            prospect,
            request,
            context,
            record,
            correction_feedback=revision_fields,
            previous_output=generated,
        )
        second_result = evaluate(corrected, 2)
        if second_result is None:
            raise ProspectResearchError(
                "The lead research correction could not be rechecked; "
                "no research assessment was saved."
            )
        final_decision, final_confidence, final_assessments, remaining_revisions = (
            assess_artifact_quality_result(
                second_result,
                overall_question_id="research_review",
                check_question_ids=check_ids,
            )
        )
        self._record_quality_decision(
            record,
            review_id=review_id,
            stage="lead.research",
            question_version=LEAD_RESEARCH_QUALITY_QUESTION_VERSION,
            pass_number=2,
            result=second_result,
            decision=final_decision,
            confidence=final_confidence,
            assessments=final_assessments,
            revision_fields=remaining_revisions,
        )
        if final_decision != "ready":
            raise ProspectResearchError(
                "The corrected lead research did not pass its Judge recheck; "
                "no research assessment was saved."
            )
        return corrected, generation.model_copy(
            update={
                "latency_ms": generation.latency_ms + correction_generation.latency_ms,
                "input_tokens": generation.input_tokens + correction_generation.input_tokens,
                "output_tokens": generation.output_tokens + correction_generation.output_tokens,
                "validation_attempts": max(
                    generation.validation_attempts, correction_generation.validation_attempts
                ),
            }
        )

    @staticmethod
    def _quality_judge_request(
        prospect: Prospect,
        generated: GeneratedAssessment,
        context: _ResearchContext,
        *,
        review_pass: int,
    ) -> JudgeEvaluationRequest:
        cited_ids = {
            citation_id for claim in generated.claim_citations for citation_id in claim.citation_ids
        }
        evidence = [
            {
                "citation_id": item.citation_id,
                "source_kind": item.source_kind,
                "title": item.title[:180],
                "source_url": str(item.source_url) if item.source_url else None,
                "excerpt": item.excerpt[:650],
            }
            for item in context.evidence
            if item.citation_id in cited_ids
        ][:16]
        qualification = (
            prospect.lead_qualification.model_dump(mode="json")
            if prospect.lead_qualification is not None
            else None
        )
        state: JsonObject = {
            "workflow": "approved_lead_deep_research",
            "question_set_version": LEAD_RESEARCH_QUALITY_QUESTION_VERSION,
            "review_pass": review_pass,
            "prospect_id": str(prospect.prospect_id),
            "deployment_id": context.deployment.deployment_id,
            "lead_qualification": qualification,
            "generated_assessment": generated.model_dump(mode="json"),
            "approved_memory_context": {
                "semantic": context.approved_memory.get("semantic", [])[:4],
                "episodic": context.approved_memory.get("episodic", [])[:3],
                "procedural": context.approved_memory.get("procedural", [])[:3],
            },
            "evidence": evidence,
        }
        revision_criteria = {
            "acceptable": "The output meets this check using the supplied evidence.",
            "needs_revision": "A specific correctable issue is supported by the supplied evidence.",
            "insufficient_evidence": (
                "The evidence is too weak or incomplete to decide; do not guess."
            ),
        }
        questions = {
            "research_review": JudgeChoiceQuestion(
                type="choice",
                instructions=(
                    "Assess whether this approved-lead research is useful and safe for a human "
                    "to review. Consider all named checks. Do not approve the lead, authorize a "
                    "proposal, infer private intent, or treat memory as public evidence."
                ),
                criteria={
                    "ready": (
                        "The research is evidence-grounded, relevant, and appropriately uncertain."
                    ),
                    "needs_revision": "At least one named check has a clear, correctable defect.",
                    "insufficient_evidence": (
                        "The supplied state does not support a reliable assessment."
                    ),
                },
            ),
            "evidence_support": JudgeChoiceQuestion(
                type="choice",
                instructions=(
                    "Check whether cited claims are supported by the exact matching evidence "
                    "items. Approved memory may explain client capabilities but is not public "
                    "evidence about this prospect."
                ),
                criteria=revision_criteria,
            ),
            "workflow_fit": JudgeChoiceQuestion(
                type="choice",
                instructions=(
                    "Check that delivery fit, workflow opportunities, and value hypothesis are "
                    "specific to observed lead signals and approved client capabilities rather "
                    "than generic sales language."
                ),
                criteria=revision_criteria,
            ),
            "uncertainty_integrity": JudgeChoiceQuestion(
                type="choice",
                instructions=(
                    "Check that missing, conflicting, or unsupported contact details, intent, "
                    "scale, internal workflow, and operational pain remain explicit unknowns "
                    "or conditional hypotheses."
                ),
                criteria=revision_criteria,
            ),
        }
        return JudgeEvaluationRequest(state=state, questions=questions)

    @staticmethod
    def _record_quality_decision(
        record: Callable[[RunEventType, JsonObject], None],
        *,
        review_id: str,
        stage: str,
        question_version: str,
        pass_number: int,
        result: JudgeEvaluationResult,
        decision: str,
        confidence: float,
        assessments: dict[str, tuple[str, float]],
        revision_fields: list[str],
    ) -> None:
        record(
            RunEventType.TOOL_COMPLETED,
            {
                "tool_name": "judge.apply_quality_policy",
                "decision_id": f"{review_id}:{stage}:quality_policy:pass_{pass_number}",
                "display_label": f"{stage} quality gate: {decision.replace('_', ' ')}",
                "question_set_version": question_version,
                "rule_version": LEAD_ARTIFACT_QUALITY_POLICY_VERSION,
                "provider": result.provider,
                "model_id": result.model_id,
                "decision": decision,
                "confidence": confidence,
                "assessments": {
                    question_id: {"decision": value[0], "confidence": value[1]}
                    for question_id, value in assessments.items()
                },
                "revision_fields": revision_fields,
                "usage": result.usage.model_dump(mode="json") if result.usage else None,
                "latency_ms": result.latency_ms,
                "external_effect_authority": "none",
            },
        )

    @staticmethod
    def memory_suggestions(result: BaseModel) -> JsonObject:
        """Turn cited research findings into conservative semantic candidates."""
        if not isinstance(result, ProspectResearchResult):
            return {}
        assessment = result.assessment
        claims = result.cited_claims
        all_references = [citation_id for claim in claims for citation_id in claim.citation_ids]
        suggestions: list[JsonObject] = []
        for signal in assessment.operational_signals[:4]:
            matching = [
                citation_id
                for claim in claims
                if signal.casefold() in claim.claim.casefold()
                for citation_id in claim.citation_ids
            ]
            suggestions.append(
                {
                    "subject": result.prospect.company_name,
                    "predicate": "has_operational_signal",
                    "value": {"statement": signal},
                    "confidence": assessment.confidence,
                    "source_references": list(dict.fromkeys(matching or all_references))[:8],
                }
            )
        return {
            "semantic": suggestions,
            "procedural": [item.model_dump(mode="json") for item in result.procedural_suggestions],
        }

    def _context_for(
        self,
        prospect: Prospect,
        request: ProspectResearchRequest,
        record: Callable[[RunEventType, JsonObject], None] | None = None,
    ) -> _ResearchContext:
        deployment = self._deployments.get(request.deployment_id)
        overlay = next(
            (
                candidate
                for candidate in deployment.agent_overlays
                if candidate.agent_id == RESEARCH_AGENT_ID and candidate.enabled
            ),
            None,
        )
        if overlay is None:
            raise RecordConflictError(
                f"Deployment '{deployment.deployment_id}' has no enabled "
                f"{RESEARCH_AGENT_ID} overlay."
            )
        allowed_collections = set(overlay.allowed_knowledge_collections)
        declared_collections = set(deployment.knowledge_collections)
        if not allowed_collections.issubset(declared_collections):
            raise ProspectResearchError(
                "The research overlay may only name collections declared by its deployment."
            )
        evidence = [
            EvidenceCitation(
                citation_id=f"prospect:{source.source_id}",
                source_kind="prospect_source",
                title=source.title,
                source_url=source.source_url,
                excerpt=source.excerpt,
            )
            for source in prospect.sources
        ]
        self._add_first_party_website_evidence(prospect, evidence, record)
        if allowed_collections:
            for document in self._knowledge.list_documents(
                deployment.deployment_id, sorted(allowed_collections)
            ):
                citation_id = f"knowledge:{document.document_id}"
                evidence.append(
                    EvidenceCitation(
                        citation_id=citation_id,
                        source_kind="knowledge",
                        title=document.title,
                        source_url=document.source_url,
                        excerpt=document.content[:4_000],
                    )
                )
            knowledge_query = f"{prospect.company_name} {request.focus}"
            for item in self._knowledge.search(
                KnowledgeSearchRequest(
                    deployment_id=deployment.deployment_id,
                    query=knowledge_query,
                    collection_keys=sorted(allowed_collections),
                    limit=8,
                )
            ):
                citation_id = f"knowledge:{item.document_id}"
                if any(candidate.citation_id == citation_id for candidate in evidence):
                    continue
                evidence.append(
                    EvidenceCitation(
                        citation_id=citation_id,
                        source_kind="knowledge",
                        title=item.title,
                        source_url=item.source_url,
                        excerpt=item.excerpt,
                    )
                )
        # Put approved client context first in the model input. This keeps the highest-priority
        # citation material visible to the small local model before the longer prospect evidence.
        evidence.sort(key=lambda item: 0 if item.source_kind == "knowledge" else 1)
        return _ResearchContext(
            deployment=deployment,
            evidence=evidence,
            approved_memory=approved_agent_memory_context(
                self._knowledge,
                deployment.deployment_id,
                RESEARCH_AGENT_ID,
                query=f"{prospect.company_name} {request.focus}",
            ),
        )

    def _add_first_party_website_evidence(
        self,
        prospect: Prospect,
        evidence: list[EvidenceCitation],
        record: Callable[[RunEventType, JsonObject], None] | None,
    ) -> None:
        """Enrich approved-lead research with bounded first-party pages when available.

        Discovery already uses the bounded website collector. Reusing it here gives the profile
        pass real service/contact pages instead of asking the model to infer a company profile from
        a search-result snippet. Crawled pages stay as assessment evidence; the prospect queue is
        not mutated by this enrichment step.
        """
        if self._website_evidence is None:
            return
        seed_url = self._first_party_website_seed(prospect)
        if seed_url is None:
            return

        invocation_id = str(uuid4())
        if record:
            record(
                RunEventType.TOOL_REQUESTED,
                {
                    "invocation_id": invocation_id,
                    "tool_name": "website_evidence",
                    "arguments": {"url": seed_url, "purpose": "approved_lead_profile"},
                },
            )
        started = perf_counter()
        try:
            pages = self._website_evidence.collect(seed_url)
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
            return
        except Exception as error:  # pragma: no cover - defensive boundary around provider code
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
            return

        existing_urls = {str(item.source_url).rstrip("/").casefold() for item in evidence}
        appended_urls: list[str] = []
        for page in pages:
            page_url = str(page.url)
            normalized_url = page_url.rstrip("/").casefold()
            if normalized_url in existing_urls:
                continue
            existing_urls.add(normalized_url)
            citation_id = "prospect:web:" + sha256(page_url.encode("utf-8")).hexdigest()[:20]
            evidence.append(
                EvidenceCitation(
                    citation_id=citation_id,
                    source_kind="prospect_source",
                    title=page.title,
                    source_url=page.url,
                    excerpt=page.description,
                )
            )
            appended_urls.append(page_url)
        if record:
            record(
                RunEventType.TOOL_COMPLETED,
                {
                    "invocation_id": invocation_id,
                    "tool_name": "website_evidence",
                    "result": {
                        "page_count": len(appended_urls),
                        "source_urls": appended_urls,
                    },
                    "latency_ms": round((perf_counter() - started) * 1000),
                },
            )

    @staticmethod
    def _first_party_website_seed(prospect: Prospect) -> str | None:
        excluded_types = {"public_company_registry", "public_procurement_ocds"}
        for source in prospect.sources:
            if source.source_type.casefold() in excluded_types:
                continue
            host = (urlparse(str(source.source_url)).hostname or "").casefold()
            if host.endswith("supercias.gob.ec"):
                continue
            return str(source.source_url)
        if prospect.website:
            website = str(prospect.website)
            host = (urlparse(website).hostname or "").casefold()
            if host and not host.endswith("supercias.gob.ec"):
                return website
        return None

    def _generate(
        self,
        prospect: Prospect,
        request: ProspectResearchRequest,
        context: _ResearchContext,
        record: Callable[[RunEventType, JsonObject], None] | None = None,
        *,
        correction_feedback: list[str] | None = None,
        previous_output: GeneratedAssessment | None = None,
    ) -> tuple[GeneratedAssessment, ResearchGenerationEvidence]:
        overlay = next(
            candidate
            for candidate in context.deployment.agent_overlays
            if candidate.agent_id == RESEARCH_AGENT_ID and candidate.enabled
        )
        profile = context.deployment.model_profile_for(overlay)
        profile_pass, profile_generation = self._generate_pass(
            prospect=prospect,
            request=request,
            context=context,
            instruction_appendix=overlay.instruction_appendix,
            output_model=ProfileResearchPass,
            operation="prospect_research.profile",
            focus=(
                "Extract a compact business profile from supplied evidence: industry, cautious "
                "size category, short summary, public contact channels, and named contact people. "
                "Always return a business_profile object. Use industry=unknown, "
                "size_category=unknown, a concise evidence-limited summary, and empty contact "
                "lists when the supplied evidence does not support those fields."
            ),
            record=record,
            correction_feedback=correction_feedback,
            previous_output=previous_output,
        )
        assessment_pass, assessment_generation = self._generate_pass(
            prospect=prospect,
            request=request,
            context=context,
            instruction_appendix=overlay.instruction_appendix,
            output_model=AssessmentResearchPass,
            operation="prospect_research.assessment",
            focus=(
                "Assess delivery fit, operational signals, workflow opportunities, value, "
                "competition, and unknowns. The value_hypothesis must connect a target signal to "
                "an approved client capability and cite one knowledge:* and one prospect:* ID in "
                "the same claim_citations entry."
            ),
            record=record,
            correction_feedback=correction_feedback,
            previous_output=previous_output,
        )
        claims = self._merge_claims(
            [*profile_pass.claim_citations, *assessment_pass.claim_citations]
        )
        generated = GeneratedAssessment(
            **assessment_pass.model_dump(exclude={"claim_citations"}),
            claim_citations=claims,
            business_profile=profile_pass.business_profile,
        )
        self._validated_reference_ids(generated, context.evidence)
        return generated, ResearchGenerationEvidence(
            deployment_id=context.deployment.deployment_id,
            agent_id=RESEARCH_AGENT_ID,
            model_id=profile.model_id,
            latency_ms=profile_generation.latency_ms + assessment_generation.latency_ms,
            input_tokens=profile_generation.input_tokens + assessment_generation.input_tokens,
            output_tokens=profile_generation.output_tokens + assessment_generation.output_tokens,
            # Keep this field within its existing per-call contract; the Run events expose the
            # individual attempts for each focused pass.
            validation_attempts=max(
                profile_generation.validation_attempts, assessment_generation.validation_attempts
            ),
        )

    def _generate_pass(
        self,
        *,
        prospect: Prospect,
        request: ProspectResearchRequest,
        context: _ResearchContext,
        instruction_appendix: str,
        output_model: type[BaseModel],
        operation: str,
        focus: str,
        record: Callable[[RunEventType, JsonObject], None] | None,
        correction_feedback: list[str] | None = None,
        previous_output: GeneratedAssessment | None = None,
    ) -> tuple[BaseModel, ResearchGenerationEvidence]:
        prompt = self._pass_prompt(
            prospect,
            request,
            context.evidence,
            instruction_appendix,
            context.approved_memory,
            output_model,
            focus,
            correction_feedback,
            previous_output,
        )
        errors: list[str] = []
        for attempt in range(1, self._settings.validation_attempts + 1):
            if record:
                record(RunEventType.MODEL_STARTED, {"operation": operation, "attempt": attempt})
            generation_prompt = prompt
            if errors:
                generation_prompt += (
                    f"\n\nPrevious response was invalid: {errors[-1]}. "
                    "Return only a concise corrected JSON object."
                )
            try:
                response = self._model_provider.generate(
                    GenerationRequest(
                        system_prompt=(
                            "You are a careful B2B research analyst. Return exactly one JSON "
                            "object, no markdown or commentary. Do not invent facts."
                        ),
                        user_prompt=generation_prompt,
                        operation=operation,
                        agent_id=RESEARCH_AGENT_ID,
                    )
                )
                if record:
                    record(
                        RunEventType.MODEL_COMPLETED,
                        {
                            "operation": operation,
                            "attempt": attempt,
                            "latency_ms": response.latency_ms,
                            "input_tokens": response.input_tokens,
                            "output_tokens": response.output_tokens,
                        },
                    )
                parsed = parse_and_validate_object(
                    response.content, output_model.model_json_schema()
                )
                value = output_model.model_validate(parsed)
                if isinstance(value, ProfileResearchPass):
                    self._validated_profile_pass(value, context.evidence)
                elif isinstance(value, AssessmentResearchPass):
                    self._validated_assessment_pass(value, context.evidence)
                return value, ResearchGenerationEvidence(
                    deployment_id=context.deployment.deployment_id,
                    agent_id=RESEARCH_AGENT_ID,
                    model_id=context.deployment.model_profile_for(
                        next(
                            item
                            for item in context.deployment.agent_overlays
                            if item.agent_id == RESEARCH_AGENT_ID and item.enabled
                        )
                    ).model_id,
                    latency_ms=response.latency_ms,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    validation_attempts=attempt,
                )
            except ModelOutputTruncatedError as error:
                message = (
                    "The previous response reached the output limit. Return a concise JSON "
                    "object with complete required fields and no extra commentary."
                )
                errors.append(message)
                self._record_model_failure(
                    record,
                    operation,
                    attempt,
                    "model_output_truncated",
                    str(error),
                    metrics=error.metrics(),
                )
            except (ContractValidationError, ValueError) as error:
                errors.append(str(error))
                self._record_model_failure(
                    record, operation, attempt, "model_output_invalid", str(error)
                )
            except BudgetExceeded as error:
                self._record_model_failure(
                    record, operation, attempt, "budget_exceeded", str(error)
                )
                raise
            except ModelProviderError as error:
                self._record_model_failure(
                    record, operation, attempt, "model_execution_failed", str(error)
                )
                raise ProspectResearchError("The configured local model is unavailable.") from error
        detail = errors[-1] if errors else "unknown validation error"
        raise ProspectResearchError(
            f"{operation} output did not satisfy the contract after "
            f"{self._settings.validation_attempts} attempts: {detail}"
        )

    @staticmethod
    def _record_model_failure(
        record: Callable[[RunEventType, JsonObject], None] | None,
        operation: str,
        attempt: int,
        code: str,
        message: str,
        *,
        metrics: dict[str, int | float] | None = None,
    ) -> None:
        if record:
            record(
                RunEventType.MODEL_FAILED,
                {
                    "operation": operation,
                    "attempt": attempt,
                    "code": code,
                    "message": message,
                    **(metrics or {}),
                },
            )

    @staticmethod
    def _pass_prompt(
        prospect: Prospect,
        request: ProspectResearchRequest,
        evidence: list[EvidenceCitation],
        instruction_appendix: str,
        approved_memory: JsonObject,
        output_model: type[BaseModel],
        focus: str,
        correction_feedback: list[str] | None = None,
        previous_output: GeneratedAssessment | None = None,
    ) -> str:
        evidence_payload = [item.model_dump(mode="json") for item in evidence]
        for item in evidence_payload:
            item["excerpt"] = str(item.get("excerpt") or "")[:900]
        knowledge_ids = [item.citation_id for item in evidence if item.source_kind == "knowledge"]
        prospect_ids = [
            item.citation_id for item in evidence if item.source_kind == "prospect_source"
        ]
        return (
            f"Research target: {prospect.company_name}, {prospect.city}, {prospect.country}.\n"
            f"Research focus: {request.focus}.\n"
            f"Task: {focus}\n"
            f"Approved client-context IDs: {knowledge_ids or 'none'}\n"
            f"Prospect-evidence IDs: {prospect_ids or 'none'}\n"
            f"Evidence: {json.dumps(evidence_payload, ensure_ascii=False)}\n"
            "Approved agent memory (context only): "
            f"{json.dumps(approved_memory, ensure_ascii=False)}\n"
            "Output JSON Schema: "
            f"{json.dumps(output_model.model_json_schema(), ensure_ascii=False)}\n"
            "Copy citation IDs exactly. Use empty lists and an explicit unknown business_profile "
            "object when evidence is insufficient; never return null for business_profile. "
            "Every URL in a profile must appear in supplied evidence. Never guess contacts or copy "
            "placeholder text such as 'protected from spam bots', 'enable JavaScript', 'not "
            "available', or 'unknown' as a contact value; omit those points and use an empty "
            "list.\n"
            + (
                "A Judge review found these correctable quality checks: "
                + ", ".join(correction_feedback)
                + ". Re-examine the same supplied evidence and revise only what it supports; "
                "leave unsupported details unknown. Previous agent output: "
                + json.dumps(previous_output.model_dump(mode="json"), ensure_ascii=False)
                + "\n"
                if correction_feedback and previous_output is not None
                else ""
            )
            + f"Client instruction appendix: {instruction_appendix or 'None.'}"
        )

    @classmethod
    def _validated_profile_pass(
        cls, value: ProfileResearchPass, evidence: list[EvidenceCitation]
    ) -> None:
        allowed = {item.citation_id for item in evidence}
        references = {
            reference for claim in value.claim_citations for reference in claim.citation_ids
        }
        if not references.issubset(allowed):
            raise ValueError("The profile pass cited source identifiers that were not provided.")
        profile_urls = cls._profile_source_urls(value.business_profile)
        provided_urls = {str(item.source_url) for item in evidence if item.source_url}
        if not profile_urls.issubset(provided_urls):
            raise ValueError("The profile pass cited URLs that were not provided to the model.")

    @staticmethod
    def _validated_assessment_pass(
        value: AssessmentResearchPass, evidence: list[EvidenceCitation]
    ) -> None:
        allowed = {item.citation_id for item in evidence}
        claim_references = {
            reference for claim in value.claim_citations for reference in claim.citation_ids
        }
        if not claim_references.issubset(allowed):
            raise ValueError("The assessment pass cited source identifiers that were not provided.")
        knowledge_ids = {item.citation_id for item in evidence if item.source_kind == "knowledge"}
        prospect_ids = {
            item.citation_id for item in evidence if item.source_kind == "prospect_source"
        }
        if not any(
            knowledge_ids.intersection(claim.citation_ids)
            and prospect_ids.intersection(claim.citation_ids)
            for claim in value.claim_citations
        ):
            raise ValueError(
                "The assessment pass must cite one approved client-context ID and one "
                "prospect-evidence ID in the same claim."
            )
        unsupported_suggestions = {
            reference
            for suggestion in value.procedural_suggestions
            for reference in suggestion.supporting_evidence
            if reference not in allowed
        }
        if unsupported_suggestions:
            raise ValueError(
                "Procedural memory suggestions must cite supplied evidence: "
                + ", ".join(sorted(unsupported_suggestions))
            )

    @staticmethod
    def _merge_claims(claims: list[ClaimCitation]) -> list[ClaimCitation]:
        merged: list[ClaimCitation] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()
        for claim in claims:
            key = (claim.claim, tuple(claim.citation_ids))
            if key in seen:
                continue
            seen.add(key)
            merged.append(claim)
        return merged[:24]

    @classmethod
    def _validated_reference_ids(
        cls, generated: GeneratedAssessment, evidence: list[EvidenceCitation]
    ) -> set[str]:
        allowed_ids = {citation.citation_id for citation in evidence}
        referenced_ids = {
            citation_id
            for cited_claim in generated.claim_citations
            for citation_id in cited_claim.citation_ids
        }
        referenced_ids.update(
            reference
            for suggestion in generated.procedural_suggestions
            for reference in suggestion.supporting_evidence
        )
        unknown_ids = referenced_ids - allowed_ids
        if unknown_ids:
            raise ValueError("The model cited source identifiers that were not provided to it.")
        if not referenced_ids:
            raise ValueError("The model returned no source-backed claims.")
        knowledge_ids = {
            citation.citation_id for citation in evidence if citation.source_kind == "knowledge"
        }
        if knowledge_ids and not (referenced_ids & knowledge_ids):
            raise ValueError(
                "The assessment did not cite the approved client context behind its "
                "value proposition."
            )
        profile_urls = cls._profile_source_urls(generated.business_profile)
        provided_urls = {str(item.source_url) for item in evidence if item.source_url}
        if not profile_urls.issubset(provided_urls):
            raise ValueError(
                "The business profile cited source URLs that were not provided to the model."
            )
        return referenced_ids

    @staticmethod
    def _profile_source_urls(profile: BusinessProfile | None) -> set[str]:
        if profile is None:
            return set()
        urls = {str(url) for point in profile.contact_points for url in point.source_urls}
        urls.update(str(url) for person in profile.contact_people for url in person.source_urls)
        urls.update(
            str(url)
            for person in profile.contact_people
            for point in person.contact_points
            for url in point.source_urls
        )
        return urls

    @staticmethod
    def _prompt(
        prospect: Prospect,
        request: ProspectResearchRequest,
        evidence: list[EvidenceCitation],
        instruction_appendix: str,
        approved_memory: JsonObject | None = None,
        compact: bool = False,
    ) -> str:
        evidence_payload = [item.model_dump(mode="json") for item in evidence]
        if compact:
            for item in evidence_payload:
                item["excerpt"] = str(item.get("excerpt") or "")[:600]
        evidence_json = json.dumps(evidence_payload, ensure_ascii=False)
        knowledge_ids = [item.citation_id for item in evidence if item.source_kind == "knowledge"]
        prospect_ids = [
            item.citation_id for item in evidence if item.source_kind == "prospect_source"
        ]
        schema = json.dumps(GeneratedAssessment.model_json_schema(), ensure_ascii=False)
        memory_json = json.dumps(
            approved_memory or {"semantic": [], "procedural": []}, ensure_ascii=False
        )
        compact_example = ""
        if compact and knowledge_ids and prospect_ids:
            compact_example = (
                "Minimal valid shape (copy the real IDs above): "
                + json.dumps(
                    {
                        "delivery_fit": "pilot",
                        "capacity_risk": "unknown",
                        "operational_signals": ["one evidence-backed signal"],
                        "workflow_opportunities": ["one bounded opportunity"],
                        "competitors": [],
                        "value_hypothesis": "one concise client-context value hypothesis",
                        "known_unknowns": [],
                        "confidence": 50,
                        "claim_citations": [
                            {
                                "claim": "one concise claim",
                                "citation_ids": [knowledge_ids[0], prospect_ids[0]],
                            }
                        ],
                        "procedural_suggestions": [],
                        "business_profile": {
                            "industry": "unknown",
                            "size_category": "unknown",
                            "summary": (
                                "Public evidence is insufficient for a fuller company profile."
                            ),
                            "contact_points": [],
                            "contact_people": [],
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        return (
            "Research target: "
            f"{prospect.company_name}, {prospect.city}, {prospect.country}.\n"
            f"Research focus: {request.focus}.\n"
            "Evidence roles: knowledge citation IDs (approved client context): "
            f"{knowledge_ids or 'none'}; "
            f"prospect citation IDs: {prospect_ids or 'none'}.\n"
            "REQUIRED CITATION CHECK: copy IDs exactly from the lists above. At least one "
            "claim_citations entry must contain one knowledge:* ID and one prospect:* ID. "
            'Example shape (use real IDs, not these placeholders): {"claim": "...", '
            '"citation_ids": ["knowledge:<approved-context-id>", '
            '"prospect:<company-source-id>"]}.\n'
            f"Evidence: {evidence_json}\n"
            f"Approved agent memory (context only; do not treat as new evidence): {memory_json}\n"
            f"Output JSON Schema: {schema}\n"
            "Use evidence only for factual statements. The value_hypothesis must connect a target "
            "signal to a specific approved client capability and cite both kinds of source in "
            "the same claim_citations entry. Never replace citation IDs with titles or URLs. "
            f"{compact_example}"
            "A competitor relationship is unknown "
            "unless an evidence URL proves it. Keep recommendations conditional and suitable "
            "for a human review. Always return a business_profile object built from the supplied "
            "evidence: identify the "
            "industry, cautious size category, concise company summary, public contact points "
            "(email, phone, WhatsApp, website, or other channels), and named contact people and "
            "roles when published. Every contact point and person must include source_urls from "
            "the supplied evidence; never guess a person or contact detail. Use empty contact "
            "lists, industry=unknown, and size_category=unknown when the evidence does not support "
            "them; the summary must still state that the public evidence is insufficient. Do not "
            "copy anti-bot or unavailable placeholders such as 'protected from spam bots', 'enable "
            "JavaScript', 'not available', or 'unknown' as contact values. Approved "
            "Keep strings concise (under 24 words where possible), use no more than 3 signals, 3 "
            "workflow opportunities, 3 claims, and 3 unknowns. "
            "procedural memory may inform how you organize the work, "
            "but never overrides the deployment instructions or evidence-citation rules. If the "
            "evidence supports a repeatable improvement to how this "
            "agent should handle future work, include it in procedural_suggestions with the "
            "supporting citation IDs; otherwise return an empty list.\n"
            f"Client instruction appendix: {instruction_appendix or 'None.'}"
        )
