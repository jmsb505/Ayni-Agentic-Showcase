"""Versioned, organization-owned Company Brain profile inputs."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from app.domain import Run

BrainText = Annotated[str, Field(min_length=1, max_length=600)]


class CompanyBrainEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=160)
    url: HttpUrl | None = None
    notes: str = Field(default="", max_length=1_000)


class CompanyBrainBaseline(BaseModel):
    """Structured, human-supplied context; brand assets are deliberately absent."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    overview: str = Field(default="", max_length=4_000)
    offerings: list[BrainText] = Field(default_factory=list, max_length=12)
    customer_segments: list[BrainText] = Field(default_factory=list, max_length=12)
    operating_regions: list[BrainText] = Field(default_factory=list, max_length=12)
    differentiators: list[BrainText] = Field(default_factory=list, max_length=12)
    capabilities: list[BrainText] = Field(default_factory=list, max_length=12)
    constraints: list[BrainText] = Field(default_factory=list, max_length=12)
    target_markets: list[BrainText] = Field(default_factory=list, max_length=12)
    supporting_evidence: list[CompanyBrainEvidence] = Field(default_factory=list, max_length=12)

    def has_research_anchor(self) -> bool:
        """Sparse founder context is enough; research discovers the rest."""
        return bool(self.overview.strip() or any(item.strip() for item in self.offerings))

    @classmethod
    def from_value(cls, value: object) -> CompanyBrainBaseline:
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            return cls()
        return cls.model_validate(value)


class CompanyBrainOrganization(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=200)
    website: str | None = Field(default=None, max_length=255)
    revision: int = Field(ge=0)


class CompanyBrainTenantRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(min_length=1, max_length=200)
    deployment_id: str = Field(min_length=1, max_length=200)


class CompanyBrainActorRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, max_length=200)
    role: Literal["owner", "admin", "member", "service"]
    kind: Literal["human", "operator"]
    capability: Literal["company.brain.research"]


class CompanyBrainResearchStart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: Literal["company_brain.v2"] = "company_brain.v2"
    kind: Literal["start_company_brain_research"] = "start_company_brain_research"
    judge_mode: Literal["off", "observe", "assist", "autonomous"] = "off"
    correlation_id: str = Field(min_length=1, max_length=200)
    operation_key: str = Field(min_length=1, max_length=200)
    tenant: CompanyBrainTenantRef
    actor: CompanyBrainActorRef
    organization: CompanyBrainOrganization
    baseline: CompanyBrainBaseline


class CompanyBrainResearchAccepted(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1, max_length=200)
    status: str = Field(min_length=1, max_length=40)


def company_brain_market_context(request: CompanyBrainResearchStart) -> dict[str, object]:
    """Build bounded research context, with stable citations for supplied facts."""
    baseline = request.baseline.model_dump(mode="json", exclude_none=True)
    source_documents: list[dict[str, str]] = []
    for field in (
        "overview",
        "offerings",
        "customer_segments",
        "operating_regions",
        "differentiators",
        "capabilities",
        "constraints",
        "target_markets",
    ):
        value = baseline.get(field)
        values = [value] if isinstance(value, str) else value if isinstance(value, list) else []
        for index, item in enumerate(values):
            text = str(item).strip()
            if text:
                source_documents.append(
                    {
                        "document_id": (
                            f"company-brain:{request.organization.id}:"
                            f"r{request.organization.revision}:{field}:{index + 1}"
                        ),
                        "content": text[:1_500],
                    }
                )
    for index, item in enumerate(request.baseline.supporting_evidence):
        source_documents.append(
            {
                "document_id": f"company-brain:{request.organization.id}:evidence:{index + 1}",
                "content": " ".join(
                    part for part in (item.label, item.notes, str(item.url or "")) if part
                )[:1_500],
            }
        )
    canonical = json.dumps(
        {
            "contract_version": request.contract_version,
            "judge_mode": request.judge_mode,
            "organization": request.organization.model_dump(mode="json", exclude_none=True),
            "baseline": baseline,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "scope": "company_brain",
        "company_brain_id": request.organization.id,
        "company_brain_revision": request.organization.revision,
        "deployment_id": request.tenant.deployment_id,
        "company_name": request.organization.name,
        "company_domain": request.organization.website,
        "company_baseline": baseline,
        "judge_mode": request.judge_mode,
        "context_fingerprint": fingerprint,
        "source_document_ids": [item["document_id"] for item in source_documents[:12]],
        "approved_offering": source_documents[:12],
    }


def company_market_profile_run_is_bound(
    run: Run,
    profile_value: dict[str, object],
    company_id: str,
    deployment_id: str,
    expected_brain_revision: int | None = None,
) -> bool:
    """Accept a profile only when its approved memory and completed Run share one subject."""
    if (
        run.agent_id != "market.research"
        or run.status.value != "completed"
        or run.input.get("deployment_id") != deployment_id
        or profile_value.get("market_research_run_id") != str(run.run_id)
    ):
        return False

    company_brain = run.input.get("company_brain")
    if isinstance(company_brain, dict):
        revision = company_brain.get("revision")
        output = run.output if isinstance(run.output, dict) else {}
        return (
            company_brain.get("organization_id") == company_id
            and isinstance(revision, int)
            and not isinstance(revision, bool)
            and revision == expected_brain_revision
            and profile_value.get("company_brain_id") == company_id
            and profile_value.get("company_brain_revision") == revision
            and output.get("scope") == "company_brain"
            and output.get("company_brain_id") == company_id
            and output.get("company_brain_revision") == revision
        )

    crm_input = run.input.get("crm")
    crm_entity = crm_input.get("entity") if isinstance(crm_input, dict) else None
    return (
        isinstance(crm_entity, dict)
        and crm_entity.get("kind") == "company"
        and crm_entity.get("id") == company_id
    )
