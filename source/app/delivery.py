"""Approval-bound outbound delivery for proposal messages."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.prospects import ProposalDraft, Prospect


class OutreachApprovalRequest(BaseModel):
    deployment_id: str = Field(min_length=1, max_length=80)
    channel: Literal["email", "whatsapp", "sms", "webhook"]
    recipient: str = Field(min_length=3, max_length=300)
    subject: str = Field(default="", max_length=240)
    body: str = Field(min_length=1, max_length=4_000)

    @field_validator("recipient", "subject", "body")
    @classmethod
    def trim(cls, value: str) -> str:
        return value.strip()


def contact_values(prospect: Prospect) -> set[tuple[str, str]]:
    profile = prospect.assessment.business_profile if prospect.assessment else None
    if profile is None:
        return set()
    points = list(profile.contact_points)
    for person in profile.contact_people:
        points.extend(person.contact_points)
    return {(point.channel.casefold(), point.value.strip()) for point in points}


def validate_outreach(
    prospect: Prospect, proposal: ProposalDraft, request: OutreachApprovalRequest
) -> None:
    if proposal.prospect_id != prospect.prospect_id:
        raise ValueError("Proposal does not belong to the selected prospect.")
    if proposal.status not in {"review_required", "approved"}:
        raise ValueError("This proposal is not available for outreach approval.")
    if request.deployment_id != prospect.deployment_id:
        raise ValueError("The proposal belongs to another deployment.")
    if (request.channel, request.recipient) not in contact_values(prospect):
        raise ValueError("The recipient must match a contact retained by approved research.")


def outreach_payload(
    prospect: Prospect, proposal: ProposalDraft, request: OutreachApprovalRequest
) -> dict:
    return {
        "proposal_id": str(proposal.proposal_id),
        "prospect_id": str(prospect.prospect_id),
        "channel": request.channel,
        "recipient": request.recipient,
        "subject": request.subject,
        "body": request.body,
    }
