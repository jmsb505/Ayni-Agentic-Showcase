from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from pathlib import Path
from threading import Lock
from uuid import uuid4

import httpx
from pydantic import ValidationError

from app.connectors.contracts import EffectClass, ErrorCategory, Retryability
from app.connectors.ports import (
    JudgeChoiceQuestion,
    JudgeEvaluationRequest,
    JudgeEvaluationResult,
    JudgeNoulQuestion,
    JudgeScoreQuestion,
    JudgeTokenUsage,
)
from app.connectors.runtime import ConnectorContext, active_connector_context
from app.tenant_context import TenantViolation

TYPESAFE_JEV_CAPSULE_ID = "ayni.judge.typesafe-jev"
TYPESAFE_JEV_PORT = "judge.evaluate"
TYPESAFE_JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MAX_STATE_BYTES = 65_536
MAX_JEV_REQUEST_BYTES = 128 * 1024
# Local-test reserve is padded above the provider's currently published input rate;
# re-check TypeSafe pricing before changing this value.
JEV_INPUT_USD_PER_MILLION_RESERVED = Decimal("0.05")
JEV_INPUT_TOKEN_SAFETY_MULTIPLIER = 4
JEV_HIDDEN_INPUT_TOKEN_RESERVE = 4_096
logger = logging.getLogger(__name__)


class JevJudgeError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        category: ErrorCategory = ErrorCategory.DEPENDENCY_UNAVAILABLE,
    ) -> None:
        self.code = code
        self.connector_error_category = category
        self.connector_retryability = Retryability.NEVER
        super().__init__(message)


class TypeSafeJevHttpProvider:
    """HTTP implementation of the provider-neutral typed judge contract."""

    def __init__(self) -> None:
        raw_budget = os.getenv("TYPESAFE_JEV_BUDGET_USD", "").strip()
        raw_max_calls = os.getenv("TYPESAFE_JEV_MAX_CALLS", "").strip()
        self._budget_limit: Decimal | None = None
        self._max_calls_limit: int | None = None
        self._budget_error: JevJudgeError | None = None
        if raw_budget:
            try:
                self._budget_limit = Decimal(raw_budget)
                if not self._budget_limit.is_finite() or self._budget_limit <= 0:
                    raise InvalidOperation
            except InvalidOperation:
                self._budget_error = JevJudgeError(
                    "JEV_BUDGET_CONFIG_INVALID",
                    "The local Jev spend cap must be a positive USD amount.",
                    category=ErrorCategory.INVALID_REQUEST,
                )
        if raw_max_calls:
            try:
                self._max_calls_limit = int(raw_max_calls)
                if self._max_calls_limit <= 0:
                    raise ValueError
            except ValueError:
                self._budget_error = JevJudgeError(
                    "JEV_BUDGET_CONFIG_INVALID",
                    "The local Jev request limit must be a positive integer.",
                    category=ErrorCategory.INVALID_REQUEST,
                )
        raw_ledger = os.getenv("TYPESAFE_JEV_BUDGET_LEDGER", "").strip()
        self._budget_ledger_path = Path(raw_ledger) if raw_ledger else None
        self._budget_lock = Lock()
        self._memory_budget_state: dict[str, object] | None = None

    def evaluate(
        self,
        request: JudgeEvaluationRequest,
        *,
        model_id: str,
        api_key: str,
        timeout_seconds: float,
    ) -> JudgeEvaluationResult:
        try:
            state_json = json.dumps(request.state, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as error:
            raise JevJudgeError(
                "JEV_INVALID_STATE",
                "Judge state must contain JSON-compatible values.",
                category=ErrorCategory.INVALID_REQUEST,
            ) from error
        if len(state_json.encode("utf-8")) > MAX_STATE_BYTES:
            raise JevJudgeError(
                "JEV_STATE_TOO_LARGE",
                "Judge state exceeds the 64 KiB request limit.",
                category=ErrorCategory.INVALID_REQUEST,
            )

        body = request.model_dump(mode="json")
        body["model"] = model_id
        body_size = len(json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        if body_size > MAX_JEV_REQUEST_BYTES:
            raise JevJudgeError(
                "JEV_REQUEST_TOO_LARGE",
                "Judge request exceeds the 128 KiB request limit.",
                category=ErrorCategory.INVALID_REQUEST,
            )
        self._reserve_budget(body_size)
        started = time.perf_counter()
        try:
            with httpx.Client(
                timeout=timeout_seconds,
                trust_env=False,
                follow_redirects=False,
            ) as client:
                response = client.post(
                    TYPESAFE_JEV_ENDPOINT,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
        except httpx.TimeoutException as error:
            raise JevJudgeError(
                "JEV_TIMEOUT", "TypeSafe Jev did not respond before the request deadline."
            ) from error
        except httpx.RequestError as error:
            raise JevJudgeError("JEV_UNAVAILABLE", "TypeSafe Jev could not be reached.") from error

        if response.status_code in {401, 403}:
            raise JevJudgeError(
                "JEV_AUTHENTICATION_FAILED",
                "TypeSafe rejected the configured API credential.",
                category=ErrorCategory.UNAUTHENTICATED,
            )
        if response.status_code == 429:
            raise JevJudgeError(
                "JEV_RATE_LIMITED",
                "TypeSafe rate-limited the judge request.",
                category=ErrorCategory.RATE_LIMITED,
            )
        if response.status_code >= 400:
            raise JevJudgeError(
                "JEV_REQUEST_FAILED",
                "TypeSafe did not accept the judge request.",
            )

        try:
            payload = response.json()
            result = self._result(request, payload, model_id, time.perf_counter() - started)
            try:
                self._record_usage(result)
            except JevJudgeError as error:
                # The conservative reservation was persisted before the request;
                # failure to append provider-reported usage must not discard a valid
                # decision or release that reservation.
                logger.error("Could not persist Jev usage details (%s).", error.code)
            return result
        except JevJudgeError:
            raise
        except (TypeError, ValueError, ValidationError) as error:
            raise JevJudgeError(
                "JEV_INVALID_RESPONSE",
                "TypeSafe returned a response that does not match the judge contract.",
            ) from error

    def _reserve_budget(self, body_size: int) -> None:
        if self._budget_error is not None:
            raise self._budget_error
        if self._budget_limit is None:
            return

        # Reserve before network I/O. The byte-derived token ceiling, extra hidden
        # prompt allowance, and padded rate intentionally overestimate usage. A
        # reservation remains charged after timeouts or malformed responses because
        # the provider may have processed the request before the failure was observed.
        token_bound = max(
            JEV_HIDDEN_INPUT_TOKEN_RESERVE,
            body_size * JEV_INPUT_TOKEN_SAFETY_MULTIPLIER + JEV_HIDDEN_INPUT_TOKEN_RESERVE,
        )
        reservation = (
            Decimal(token_bound)
            * JEV_INPUT_USD_PER_MILLION_RESERVED
            / Decimal(1_000_000)
        ).quantize(Decimal("0.00000001"), rounding=ROUND_CEILING)

        with self._budget_lock:
            state = self._read_budget_state()
            try:
                previous_limit = Decimal(str(state.get("limit_usd", self._budget_limit)))
                reserved = Decimal(str(state.get("reserved_usd", "0")))
                calls_reserved = int(state.get("calls_reserved", 0))
                previous_max_calls_raw = state.get("max_calls")
                previous_max_calls = (
                    int(previous_max_calls_raw)
                    if previous_max_calls_raw is not None
                    else self._max_calls_limit
                )
            except InvalidOperation as error:
                raise JevJudgeError(
                    "JEV_BUDGET_LEDGER_INVALID",
                    "The local Jev spend ledger is invalid; refusing to send a request.",
                    category=ErrorCategory.INVALID_REQUEST,
                ) from error
            except (TypeError, ValueError) as error:
                raise JevJudgeError(
                    "JEV_BUDGET_LEDGER_INVALID",
                    "The local Jev spend ledger is invalid; refusing to send a request.",
                    category=ErrorCategory.INVALID_REQUEST,
                ) from error
            if (
                not previous_limit.is_finite()
                or previous_limit <= 0
                or not reserved.is_finite()
                or reserved < 0
                or calls_reserved < 0
                or (previous_max_calls is not None and previous_max_calls <= 0)
            ):
                raise JevJudgeError(
                    "JEV_BUDGET_LEDGER_INVALID",
                    "The local Jev spend ledger is invalid; refusing to send a request.",
                    category=ErrorCategory.INVALID_REQUEST,
                )
            effective_limit = min(previous_limit, self._budget_limit)
            effective_max_calls = previous_max_calls
            if self._max_calls_limit is not None:
                effective_max_calls = (
                    min(previous_max_calls, self._max_calls_limit)
                    if previous_max_calls is not None
                    else self._max_calls_limit
                )
            if effective_max_calls is not None and calls_reserved >= effective_max_calls:
                raise JevJudgeError(
                    "JEV_CALL_LIMIT_EXCEEDED",
                    "The local Jev request limit has been reached; no request was sent.",
                    category=ErrorCategory.FORBIDDEN,
                )
            if reserved + reservation > effective_limit:
                raise JevJudgeError(
                    "JEV_BUDGET_EXCEEDED",
                    "The local Jev spend cap would be exceeded; no request was sent.",
                    category=ErrorCategory.FORBIDDEN,
                )
            state["schema_version"] = 1
            state["limit_usd"] = format(effective_limit, "f")
            state["reserved_usd"] = format(reserved + reservation, ".8f")
            state["calls_reserved"] = calls_reserved + 1
            if effective_max_calls is not None:
                state["max_calls"] = effective_max_calls
            state["last_reservation_usd"] = format(reservation, ".8f")
            state["last_token_upper_bound"] = token_bound
            self._write_budget_state(state)

    def _record_usage(self, result: JudgeEvaluationResult) -> None:
        if self._budget_limit is None or result.usage is None:
            return
        with self._budget_lock:
            state = self._read_budget_state()
            state["reported_input_tokens"] = int(state.get("reported_input_tokens", 0)) + (
                result.usage.input_tokens or 0
            )
            state["reported_output_tokens"] = int(state.get("reported_output_tokens", 0)) + (
                result.usage.output_tokens or 0
            )
            state["calls_with_usage"] = int(state.get("calls_with_usage", 0)) + 1
            self._write_budget_state(state)

    def _read_budget_state(self) -> dict[str, object]:
        if self._budget_ledger_path is None:
            if self._memory_budget_state is None:
                self._memory_budget_state = {
                    "schema_version": 1,
                    "limit_usd": format(self._budget_limit or Decimal("0"), "f"),
                    "reserved_usd": "0",
                }
            return dict(self._memory_budget_state)
        if not self._budget_ledger_path.exists():
            return {
                "schema_version": 1,
                "limit_usd": format(self._budget_limit or Decimal("0"), "f"),
                "reserved_usd": "0",
            }
        try:
            state = json.loads(self._budget_ledger_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise JevJudgeError(
                "JEV_BUDGET_LEDGER_INVALID",
                "The local Jev spend ledger cannot be read; refusing to send a request.",
                category=ErrorCategory.INVALID_REQUEST,
            ) from error
        if not isinstance(state, dict) or state.get("schema_version") != 1:
            raise JevJudgeError(
                "JEV_BUDGET_LEDGER_INVALID",
                "The local Jev spend ledger is invalid; refusing to send a request.",
                category=ErrorCategory.INVALID_REQUEST,
            )
        return state

    def _write_budget_state(self, state: dict[str, object]) -> None:
        if self._budget_ledger_path is None:
            self._memory_budget_state = dict(state)
            return
        try:
            self._budget_ledger_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self._budget_ledger_path.with_name(
                f".{self._budget_ledger_path.name}.{uuid4().hex}.tmp"
            )
            temporary_path.write_text(
                json.dumps(state, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            temporary_path.replace(self._budget_ledger_path)
        except OSError as error:
            raise JevJudgeError(
                "JEV_BUDGET_LEDGER_WRITE_FAILED",
                "The local Jev spend ledger cannot be updated; refusing to send a request.",
                category=ErrorCategory.INVALID_REQUEST,
            ) from error

    @staticmethod
    def _result(
        request: JudgeEvaluationRequest,
        payload: object,
        configured_model_id: str,
        elapsed_seconds: float,
    ) -> JudgeEvaluationResult:
        if not isinstance(payload, dict):
            raise ValueError("response must be an object")
        raw_answers = payload.get("answers")
        if not isinstance(raw_answers, dict) or set(raw_answers) != set(request.questions):
            raise ValueError("response answers must match the requested questions")

        normalized_answers: dict[str, dict[str, object]] = {}
        for question_id, question in request.questions.items():
            answer = raw_answers[question_id]
            if not isinstance(answer, dict) or answer.get("type") != question.type:
                raise ValueError("response answer type does not match its question")
            normalized_answer = dict(answer)
            if isinstance(question, JudgeChoiceQuestion):
                if answer.get("choice") not in question.criteria:
                    raise ValueError("choice answer is outside the declared criteria")
                probabilities = answer.get("probabilities", {})
                if not isinstance(probabilities, dict) or set(probabilities) - set(
                    question.criteria
                ):
                    raise ValueError("choice probabilities are outside the declared criteria")
                TypeSafeJevHttpProvider._validate_probabilities(probabilities)
            elif isinstance(question, JudgeScoreQuestion):
                score = answer.get("score")
                if (
                    not isinstance(score, (float, int))
                    or isinstance(score, bool)
                    or not 0 <= score <= len(question.criteria) - 1
                ):
                    raise ValueError("score answer is outside the declared criteria")
                probabilities = answer.get("probabilities", {})
                if not isinstance(probabilities, dict):
                    raise ValueError("score probabilities must be an object")
                TypeSafeJevHttpProvider._validate_probabilities(probabilities)
                legend = answer.get("legend")
                # TypeSafe Score legends may contain structured examples, while the
                # provider-neutral v1 port intentionally exposes only string labels.
                # The criteria already travel with the request, so omit an optional
                # provider-specific legend rather than failing the whole judgment.
                if legend is not None and (
                    not isinstance(legend, dict)
                    or any(not isinstance(label, str) for label in legend.values())
                ):
                    normalized_answer.pop("legend", None)
            elif not isinstance(question, JudgeNoulQuestion):
                raise ValueError("unsupported judge question type")
            else:
                noul = answer.get("noul")
                if (
                    not isinstance(noul, (float, int))
                    or isinstance(noul, bool)
                    or not 0 <= noul <= 1
                ):
                    raise ValueError("Noul answer must be a probability from 0 to 1")

            confidence = answer.get("confidence")
            if confidence is not None and (
                not isinstance(confidence, (float, int))
                or isinstance(confidence, bool)
                or not 0 <= confidence <= 1
            ):
                raise ValueError("answer confidence must be a probability from 0 to 1")
            normalized_answers[question_id] = normalized_answer

        model_id = payload.get("model", configured_model_id)
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("response model identifier is missing")
        usage_payload = payload.get("usage")
        usage = None
        if usage_payload is not None:
            if not isinstance(usage_payload, dict):
                raise ValueError("response usage must be an object")
            usage = JudgeTokenUsage(
                input_tokens=usage_payload.get("input_tokens", usage_payload.get("prompt_tokens")),
                output_tokens=usage_payload.get(
                    "output_tokens", usage_payload.get("completion_tokens")
                ),
            )
        return JudgeEvaluationResult(
            provider="typesafe.jev",
            model_id=model_id,
            answers=normalized_answers,
            usage=usage,
            latency_ms=elapsed_seconds * 1000,
        )

    @staticmethod
    def _validate_probabilities(probabilities: dict) -> None:
        if any(
            not isinstance(value, (float, int)) or isinstance(value, bool) or not 0 <= value <= 1
            for value in probabilities.values()
        ):
            raise ValueError("answer probabilities must be numbers from 0 to 1")


class ConnectorJevJudgeAdapter:
    """Routes Jev judgments through the deployment-pinned connector port."""

    capsule_id = TYPESAFE_JEV_CAPSULE_ID
    port_name = TYPESAFE_JEV_PORT

    def __init__(
        self,
        *,
        context_factory: Callable[[list[str], str | None], ConnectorContext],
        provider: TypeSafeJevHttpProvider | None = None,
    ) -> None:
        self._context_factory = context_factory
        self._provider = provider or TypeSafeJevHttpProvider()

    def evaluate(
        self,
        request: JudgeEvaluationRequest,
        *,
        deployment_id: str | None = None,
    ) -> JudgeEvaluationResult:
        context = active_connector_context()
        if context is None:
            context = self._context_factory([self.capsule_id], deployment_id)
        elif deployment_id and deployment_id != context.deployment_id:
            raise TenantViolation("Judge request does not match the active Run deployment")

        def dispatch(port_request: JudgeEvaluationRequest) -> JudgeEvaluationResult:
            binding = context.capsule_binding(self.capsule_id)
            if not binding or binding.get("enabled") is False:
                raise JevJudgeError(
                    "JEV_NOT_ENABLED",
                    "TypeSafe Jev is not enabled for this deployment.",
                    category=ErrorCategory.FORBIDDEN,
                )
            if self.port_name not in binding.get("granted_permissions", []):
                raise JevJudgeError(
                    "JEV_PERMISSION_MISSING",
                    "The deployment has not granted permission to evaluate judge requests.",
                    category=ErrorCategory.FORBIDDEN,
                )
            secret_refs = binding.get("secret_refs", {})
            api_key_ref = secret_refs.get("api_key") if isinstance(secret_refs, dict) else None
            api_key = os.getenv(api_key_ref, "").strip() if api_key_ref else ""
            if not api_key:
                raise JevJudgeError(
                    "JEV_NOT_CONFIGURED",
                    "The TypeSafe API credential is not configured for this deployment.",
                )
            configuration = binding.get("configuration", {})
            configuration = configuration if isinstance(configuration, dict) else {}
            return self._provider.evaluate(
                port_request,
                model_id=configuration.get("model_id", "jev-latest"),
                api_key=api_key,
                timeout_seconds=configuration.get("timeout_seconds", 15),
            )

        exchange = context.invoke(
            source_id="ayni.core",
            target_id=self.capsule_id,
            port_name=self.port_name,
            payload=request,
            result_type=JudgeEvaluationResult,
            operation_id=f"{self.capsule_id}:{self.port_name}:{uuid4()}",
            effect_class=EffectClass.NONE,
            handler=dispatch,
        )
        return exchange.unwrap()
