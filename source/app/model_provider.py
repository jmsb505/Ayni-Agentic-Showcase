from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol

from app.config import Settings


class ModelProviderError(RuntimeError):
    failure_code = "model_execution_failed"


def require_gpu_headroom(
    minimum_free_mib: int,
    free_memory_bytes: int | None,
    *,
    cuda_available: bool,
) -> None:
    """Refuse model loading when a configured test GPU floor is not met."""
    if minimum_free_mib <= 0:
        return
    if not cuda_available or free_memory_bytes is None:
        raise ModelProviderError(
            f"This test requires CUDA with at least {minimum_free_mib} MiB free "
            "before loading Qwen."
        )
    minimum_free_bytes = minimum_free_mib * 1024 * 1024
    if free_memory_bytes < minimum_free_bytes:
        free_mib = free_memory_bytes // (1024 * 1024)
        raise ModelProviderError(
            f"Qwen load blocked: {free_mib} MiB GPU memory is free; "
            f"test mode requires {minimum_free_mib} MiB. Close other GPU workloads and retry."
        )


def gpu_allocator_fraction_for_reserve(
    free_memory_bytes: int,
    total_memory_bytes: int,
    reserve_mib: int,
) -> float:
    """Bound PyTorch's CUDA caching allocator while preserving a free-memory reserve."""
    if total_memory_bytes <= 0 or reserve_mib < 0:
        raise ModelProviderError("GPU memory limit configuration is invalid.")
    allocation_budget = free_memory_bytes - reserve_mib * 1024 * 1024
    if allocation_budget <= 0:
        raise ModelProviderError(
            f"Qwen load blocked: GPU headroom cannot preserve the {reserve_mib} MiB reserve."
        )
    return min(allocation_budget / total_memory_bytes, 1.0)


def generation_was_truncated(
    *,
    output_tokens: int,
    generation_limit: int,
    elapsed_seconds: float,
    time_limit_seconds: float,
    ended_by_eos: bool,
) -> bool:
    """Distinguish complete model output from generation stopped by either budget."""
    if ended_by_eos:
        return False
    return output_tokens >= generation_limit or elapsed_seconds >= time_limit_seconds


class ModelOutputTruncatedError(ModelProviderError):
    failure_code = "model_output_truncated"

    def __init__(
        self,
        message: str,
        *,
        latency_ms: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        max_new_tokens: int | None = None,
        max_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.latency_ms = latency_ms
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.max_new_tokens = max_new_tokens
        self.max_seconds = max_seconds

    def metrics(self) -> dict[str, int | float]:
        values: dict[str, int | float | None] = {
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "max_new_tokens": self.max_new_tokens,
            "max_seconds": self.max_seconds,
        }
        return {key: value for key, value in values.items() if value is not None}


@dataclass(frozen=True)
class GenerationRequest:
    system_prompt: str
    user_prompt: str
    operation: str = "structured_generation"
    # Callers with a compact contract can opt into a smaller budget without changing the
    # deployment-wide model setting. The provider never allows this override to exceed the
    # configured maximum.
    max_new_tokens: int | None = None
    max_seconds: float | None = None
    agent_id: str | None = None
    model_profile: str | None = None


@dataclass(frozen=True)
class GenerationResponse:
    content: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    truncated: bool
    provider: str | None = None
    model_id: str | None = None
    resource_id: str | None = None
    profile_name: str | None = None
    route_reason: str | None = None


class ModelProvider(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResponse: ...

    def health(self) -> dict[str, Any]: ...


class LocalQwenProvider:
    """Lazy, single-instance local Qwen provider adapted from proven SmartOps mechanics."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._load_lock = threading.Lock()
        self._generation_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._metrics: deque[dict[str, int | float | str | bool]] = deque(maxlen=1_000)
        self._load_error: str | None = None
        self._load_latency_ms: float | None = None
        self._gpu_device_index: int | None = None
        self._gpu_free_before_load_mib: int | None = None
        self._gpu_allocation_limit_mib: int | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def health(self) -> dict[str, Any]:
        return {
            "provider": "local_qwen",
            "model_id": self.settings.model_id,
            "model_path": str(self.settings.model_path),
            "max_new_tokens": self.settings.model_max_new_tokens,
            "model_files_present": self.settings.model_path.exists(),
            "loaded": self.is_loaded,
            "device": self.settings.model_device,
            "runtime": "nf4" if self.settings.model_load_in_4bit else "bf16",
            "attention_implementation": self.settings.model_attention_implementation,
            "load_latency_ms": self._load_latency_ms,
            "load_error": self._load_error,
            "gpu_device_index": self._gpu_device_index,
            "gpu_free_before_load_mib": self._gpu_free_before_load_mib,
            "gpu_allocator_limit_mib": self._gpu_allocation_limit_mib,
            "gpu_peak_allocated_mib": self.performance()["peak_gpu_allocated_mib"],
        }

    def performance(self) -> dict[str, Any]:
        with self._metrics_lock:
            records = list(self._metrics)
        total_latency = sum(float(item["latency_ms"]) for item in records)
        output_tokens = sum(int(item["output_tokens"]) for item in records)
        gpu_peaks = [
            int(item["gpu_peak_allocated_mib"])
            for item in records
            if "gpu_peak_allocated_mib" in item
        ]
        return {
            "generations": len(records),
            "input_tokens": sum(int(item["input_tokens"]) for item in records),
            "output_tokens": output_tokens,
            "latency_ms": round(total_latency, 1),
            "output_tokens_per_second": round(output_tokens / max(total_latency / 1_000, 0.001), 1),
            "truncated_generations": sum(bool(item["truncated"]) for item in records),
            "load_latency_ms": self._load_latency_ms,
            "peak_gpu_allocated_mib": max(gpu_peaks) if gpu_peaks else None,
        }

    def load(self) -> None:
        if self.is_loaded:
            return
        if not self.settings.model_path.exists():
            raise ModelProviderError(
                f"Local model files are missing at {self.settings.model_path}."
            )

        with self._load_lock:
            if self.is_loaded:
                return
            started = time.perf_counter()
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
                from transformers.utils import logging

                logging.set_verbosity_error()
                logging.disable_progress_bar()
                if self.settings.model_min_free_gpu_mib > 0:
                    cuda_available = (
                        self.settings.model_device.startswith("cuda") and torch.cuda.is_available()
                    )
                    device = torch.device(self.settings.model_device)
                    self._gpu_device_index = (
                        device.index if device.index is not None else torch.cuda.current_device()
                    ) if cuda_available else None
                    free_memory_bytes, total_memory_bytes = (
                        torch.cuda.mem_get_info(self._gpu_device_index)
                        if cuda_available and self._gpu_device_index is not None
                        else (None, None)
                    )
                    require_gpu_headroom(
                        self.settings.model_min_free_gpu_mib,
                        free_memory_bytes,
                        cuda_available=cuda_available,
                    )
                    if free_memory_bytes is None or total_memory_bytes is None:
                        raise ModelProviderError(
                            "Unable to inspect CUDA memory before loading Qwen."
                        )
                    self._gpu_free_before_load_mib = free_memory_bytes // (1024 * 1024)
                    allocator_fraction = gpu_allocator_fraction_for_reserve(
                        free_memory_bytes,
                        total_memory_bytes,
                        self.settings.model_gpu_reserve_mib,
                    )
                    try:
                        torch.cuda.set_per_process_memory_fraction(
                            allocator_fraction,
                            self._gpu_device_index,
                        )
                    except Exception as error:
                        raise ModelProviderError(
                            "Unable to enforce the TestMode CUDA allocation ceiling."
                        ) from error
                    self._gpu_allocation_limit_mib = int(
                        allocator_fraction * total_memory_bytes / (1024 * 1024)
                    )
                model_options: dict[str, Any] = {
                    "device_map": "auto",
                    "dtype": torch.bfloat16,
                    "attn_implementation": self.settings.model_attention_implementation,
                }
                if self.settings.model_load_in_4bit:
                    model_options["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                self._tokenizer = AutoTokenizer.from_pretrained(self.settings.model_path)
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.settings.model_path,
                    **model_options,
                )
                self._load_latency_ms = round(
                    (time.perf_counter() - started) * 1_000,
                    1,
                )
                self._load_error = None
            except Exception as error:  # pragma: no cover - hardware/runtime dependent
                self._model = None
                self._tokenizer = None
                self._load_latency_ms = round(
                    (time.perf_counter() - started) * 1_000,
                    1,
                )
                self._load_error = str(error)
                if isinstance(error, ModelProviderError):
                    raise
                raise ModelProviderError("Unable to load the local Qwen model.") from error

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.load()
        if not self.is_loaded:
            raise ModelProviderError("Local model did not load.")

        with self._generation_lock:
            messages = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ]
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.settings.model_context_window,
            ).to(self._model.device)
            if self._gpu_device_index is not None:
                import torch

                torch.cuda.reset_peak_memory_stats(self._gpu_device_index)
            started = time.perf_counter()
            generation_limit = self.settings.model_max_new_tokens
            if request.max_new_tokens is not None:
                generation_limit = min(generation_limit, max(1, request.max_new_tokens))
            generation_time_limit = request.max_seconds or 120
            output = self._model.generate(
                **inputs,
                max_new_tokens=generation_limit,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
                max_time=generation_time_limit,
            )
            latency_ms = (time.perf_counter() - started) * 1_000
            gpu_peak_allocated_mib = None
            if self._gpu_device_index is not None:
                gpu_peak_allocated_mib = int(
                    torch.cuda.max_memory_allocated(self._gpu_device_index) / (1024 * 1024)
                )
            generated = output[0][inputs.input_ids.shape[-1] :]
            input_tokens = int(inputs.input_ids.shape[-1])
            output_tokens = int(generated.shape[-1])
            eos_token_ids = getattr(
                getattr(self._model, "generation_config", None),
                "eos_token_id",
                None,
            )
            if eos_token_ids is None:
                eos_token_ids = self._tokenizer.eos_token_id
            if isinstance(eos_token_ids, int):
                eos_token_ids = {eos_token_ids}
            elif eos_token_ids is None:
                eos_token_ids = set()
            else:
                eos_token_ids = {int(token_id) for token_id in eos_token_ids}
            # `generated` is the first (and only) batch row from the model output,
            # so it is a one-dimensional token sequence here.
            last_token_id = int(generated[-1]) if output_tokens else None
            ended_by_eos = last_token_id in eos_token_ids
            elapsed_seconds = latency_ms / 1_000
            truncated = generation_was_truncated(
                output_tokens=output_tokens,
                generation_limit=generation_limit,
                elapsed_seconds=elapsed_seconds,
                time_limit_seconds=generation_time_limit,
                ended_by_eos=ended_by_eos,
            )
            content = self._tokenizer.decode(generated, skip_special_tokens=True)
            metric: dict[str, int | float | str | bool] = {
                "operation": request.operation,
                "latency_ms": latency_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "truncated": truncated,
            }
            if gpu_peak_allocated_mib is not None:
                metric["gpu_peak_allocated_mib"] = gpu_peak_allocated_mib
            with self._metrics_lock:
                self._metrics.append(metric)
            if truncated:
                reason = (
                    "time limit"
                    if elapsed_seconds >= generation_time_limit
                    and output_tokens < generation_limit
                    else "generation limit"
                )
                raise ModelOutputTruncatedError(
                    f"Model response reached the {reason} before completion.",
                    latency_ms=round(latency_ms, 1),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    max_new_tokens=generation_limit,
                    max_seconds=generation_time_limit,
                )
            return GenerationResponse(
                content=content,
                latency_ms=round(latency_ms, 1),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                truncated=truncated,
            )

    def unload(self) -> None:
        """Release model and CUDA resources during application shutdown."""
        with self._generation_lock:
            with self._load_lock:
                model = self._model
                self._model = None
                self._tokenizer = None

                if model is None:
                    return

                del model
                try:
                    import gc

                    import torch

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    # The local-model dependencies are optional in test/dev environments.
                    pass
