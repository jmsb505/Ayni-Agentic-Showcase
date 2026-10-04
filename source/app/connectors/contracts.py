from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any, Generic, Literal, TypeVar

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

PayloadT = TypeVar("PayloadT", bound=BaseModel | dict[str, Any])
ResultT = TypeVar("ResultT", bound=BaseModel | dict[str, Any])
WIRE_DATETIME = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})$"
)


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EffectClass(str, Enum):
    NONE = "none"
    READ = "read"
    CONSEQUENTIAL = "consequential"


class Retryability(str, Enum):
    SAFE = "safe"
    NEVER = "never"
    RECONCILE_FIRST = "reconcile_first"


class ErrorCategory(str, Enum):
    INVALID_REQUEST = "invalid_request"
    UNAUTHENTICATED = "unauthenticated"
    FORBIDDEN = "forbidden"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    RATE_LIMITED = "rate_limited"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    OUTCOME_UNKNOWN = "outcome_unknown"
    INTERNAL = "internal"


class ResponseStatus(str, Enum):
    ACCEPTED = "accepted"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


class CapsuleReference(StrictContract):
    id: str = Field(
        min_length=1,
        max_length=120,
        pattern=r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$",
    )
    version: str = Field(
        pattern=r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
    )


class PortReference(StrictContract):
    name: str = Field(
        min_length=3,
        max_length=160,
        pattern=r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$",
    )
    contract_version: str = Field(
        pattern=r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
    )


class TenantReference(StrictContract):
    tenant_id: str = Field(min_length=1, max_length=200)
    deployment_id: str = Field(min_length=1, max_length=200)


class ActorContext(StrictContract):
    subject_id: str = Field(min_length=1, max_length=200)
    kind: str = Field(pattern=r"^(human|agent|service|operator|system)$")
    issuer: str | None = Field(default=None, min_length=1, max_length=200)


class ConnectorContext(StrictContract):
    tenant: TenantReference
    actor: ActorContext | None = None
    correlation_id: str = Field(min_length=1, max_length=200)
    run_id: str | None = Field(default=None, min_length=1, max_length=200)
    deadline_at: AwareDatetime | None = None

    @field_validator("deadline_at", mode="before")
    @classmethod
    def deadline_is_wire_datetime(cls, value):
        if value is not None:
            if not isinstance(value, (str, datetime)):
                raise ValueError("deadline_at must be an ISO datetime string")
            if isinstance(value, str) and not WIRE_DATETIME.fullmatch(value):
                raise ValueError("deadline_at must include seconds and a timezone")
        return value


class OperationIdentity(StrictContract):
    operation_id: str = Field(min_length=1, max_length=200)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=200)
    attempt: int = Field(default=1, ge=1, strict=True)


class ConnectorRequest(StrictContract, Generic[PayloadT]):
    connector_version: Literal["connector.v1"]
    message_id: str = Field(min_length=1, max_length=200)
    source: CapsuleReference
    target: CapsuleReference
    port: PortReference
    context: ConnectorContext
    operation: OperationIdentity
    effect_class: EffectClass
    payload: PayloadT


class EffectResourceReference(StrictContract):
    kind: str = Field(min_length=1, max_length=200)
    id: str = Field(min_length=1, max_length=200)


class EffectReceipt(StrictContract):
    provider: str = Field(min_length=1, max_length=200)
    effect_id: str = Field(min_length=1, max_length=200)
    operation_key: str = Field(min_length=1, max_length=200)
    resource_refs: list[EffectResourceReference] = Field(default_factory=list, max_length=100)
    observed_at: AwareDatetime
    evidence_digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @field_validator("observed_at", mode="before")
    @classmethod
    def observed_at_is_wire_datetime(cls, value):
        if not isinstance(value, (str, datetime)):
            raise ValueError("observed_at must be an ISO datetime string")
        if isinstance(value, str) and not WIRE_DATETIME.fullmatch(value):
            raise ValueError("observed_at must include seconds and a timezone")
        return value


class ConnectorError(StrictContract):
    code: str = Field(min_length=1, max_length=80, pattern=r"^[A-Z][A-Z0-9_]*$")
    message: str = Field(min_length=1, max_length=1000)
    category: ErrorCategory
    retryability: Retryability
    retry_after_ms: int | None = Field(default=None, ge=0, strict=True)
    reconciliation_ref: str | None = Field(default=None, min_length=1, max_length=200)


class ConnectorResponse(StrictContract, Generic[ResultT]):
    connector_version: Literal["connector.v1"]
    message_id: str = Field(min_length=1, max_length=200)
    in_reply_to: str = Field(min_length=1, max_length=200)
    correlation_id: str = Field(min_length=1, max_length=200)
    source: CapsuleReference
    target: CapsuleReference
    port: PortReference
    effect_class: EffectClass
    status: ResponseStatus
    operation_ref: str | None = Field(default=None, min_length=1, max_length=200)
    result: ResultT | None = None
    error: ConnectorError | None = None
    receipt: EffectReceipt | None = None

    @model_validator(mode="after")
    def validate_outcome(self):
        if self.status == ResponseStatus.ACCEPTED:
            if not self.operation_ref:
                raise ValueError("accepted responses require operation_ref")
            if any(value is not None for value in (self.result, self.error, self.receipt)):
                raise ValueError("accepted responses cannot include a result, error, or receipt")
        elif self.status == ResponseStatus.SUCCEEDED:
            if self.operation_ref is not None:
                raise ValueError("succeeded responses cannot include operation_ref")
            if self.result is None:
                raise ValueError("succeeded responses require result")
            if self.error is not None:
                raise ValueError("succeeded responses cannot include error")
            if self.effect_class == EffectClass.CONSEQUENTIAL and self.receipt is None:
                raise ValueError("consequential success requires an effect receipt")
            if self.effect_class != EffectClass.CONSEQUENTIAL and self.receipt is not None:
                raise ValueError("only consequential success can include an effect receipt")
        elif self.status == ResponseStatus.FAILED:
            if self.operation_ref is not None:
                raise ValueError("failed responses cannot include operation_ref")
            if self.error is None:
                raise ValueError("failed responses require error")
            if self.error.retryability == Retryability.RECONCILE_FIRST:
                raise ValueError("known failures cannot require reconciliation before retry")
            if self.result is not None or self.receipt is not None:
                raise ValueError("failed responses cannot include a result or receipt")
        elif self.status == ResponseStatus.UNCERTAIN:
            if self.operation_ref is not None:
                raise ValueError("uncertain responses cannot include operation_ref")
            if self.error is None:
                raise ValueError("uncertain responses require error")
            if self.error.retryability != Retryability.RECONCILE_FIRST:
                raise ValueError("uncertain responses require reconcile_first retryability")
            if self.error.category != ErrorCategory.OUTCOME_UNKNOWN:
                raise ValueError("uncertain responses require the outcome_unknown category")
            if not self.error.reconciliation_ref:
                raise ValueError("uncertain responses require reconciliation_ref")
            if self.result is not None or self.receipt is not None:
                raise ValueError("uncertain responses cannot claim a result or receipt")
        return self
