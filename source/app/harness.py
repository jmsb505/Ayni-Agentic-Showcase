from __future__ import annotations

import json
from typing import Protocol

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.actions import ActionDefinition, ActionExecution
from app.domain import AgentVersion, JsonObject
from app.model_provider import GenerationRequest
from app.tools import ToolDefinition, ToolExecution


class HarnessNotFoundError(LookupError):
    pass


class HarnessAdapter(Protocol):
    def build_action_request(
        self,
        version: AgentVersion,
        run_input: JsonObject,
        actions: list[ActionDefinition],
    ) -> GenerationRequest: ...

    def build_tool_request(
        self,
        version: AgentVersion,
        run_input: JsonObject,
        tools: list[ToolDefinition],
    ) -> GenerationRequest: ...

    def build_request(
        self,
        version: AgentVersion,
        run_input: JsonObject,
        *,
        previous_output: str | None = None,
        validation_feedback: str | None = None,
        tool_execution: ToolExecution | None = None,
        action_execution: ActionExecution | None = None,
    ) -> GenerationRequest: ...


class SimpleJsonHarness:
    def build_action_request(
        self,
        version: AgentVersion,
        run_input: JsonObject,
        actions: list[ActionDefinition],
    ) -> GenerationRequest:
        available = [action.model_dump(mode="json") for action in actions]
        available_json = json.dumps(
            available,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        prompt = (
            f"Input:\n{json.dumps(run_input, separators=(',', ':'), ensure_ascii=False)}\n\n"
            f"Available actions:\n{available_json}\n\n"
            "Select exactly one available action needed to cause the requested controlled effect. "
            'Return exactly one JSON object shaped as {"action":{"action_type":"name",'
            '"payload":{...}}}. The payload must conform only to the selected input_schema; never '
            "copy output_schema fields into it. Do not include markdown or explanatory prose."
        )
        return GenerationRequest(
            system_prompt=version.instructions,
            user_prompt=prompt,
            operation=f"action_selection:{version.agent_id}@{version.version}",
        )

    def build_tool_request(
        self,
        version: AgentVersion,
        run_input: JsonObject,
        tools: list[ToolDefinition],
    ) -> GenerationRequest:
        available = [tool.model_dump(mode="json") for tool in tools]
        available_json = json.dumps(
            available,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        prompt = (
            f"Input:\n{json.dumps(run_input, separators=(',', ':'), ensure_ascii=False)}\n\n"
            f"Available tools:\n{available_json}\n\n"
            "Select exactly one available tool required to complete the task. Return exactly one "
            'JSON object shaped as {"tool_call":{"name":"tool_name","arguments":{...}}}. '
            "Do not include markdown or explanatory prose."
        )
        return GenerationRequest(
            system_prompt=version.instructions,
            user_prompt=prompt,
            operation=f"tool_selection:{version.agent_id}@{version.version}",
        )

    def build_request(
        self,
        version: AgentVersion,
        run_input: JsonObject,
        *,
        previous_output: str | None = None,
        validation_feedback: str | None = None,
        tool_execution: ToolExecution | None = None,
        action_execution: ActionExecution | None = None,
    ) -> GenerationRequest:
        schema = json.dumps(version.output_schema, separators=(",", ":"), ensure_ascii=False)
        user_input = json.dumps(run_input, separators=(",", ":"), ensure_ascii=False)
        prompt = (
            f"Input:\n{user_input}\n\n"
            f"Return exactly one JSON object conforming to this JSON Schema:\n{schema}\n\n"
            "Do not include markdown or explanatory prose."
        )
        if tool_execution is not None:
            tool_context = json.dumps(
                tool_execution.model_dump(mode="json"),
                separators=(",", ":"),
                ensure_ascii=False,
            )
            prompt += (
                "\n\nA platform-controlled tool returned this validated result. Use it when "
                f"producing the final answer:\n{tool_context}"
            )
        if action_execution is not None:
            action_context = json.dumps(
                action_execution.model_dump(mode="json"),
                separators=(",", ":"),
                ensure_ascii=False,
            )
            prompt += (
                "\n\nA platform-controlled action produced this validated and verified result. "
                f"Use it when producing the final answer:\n{action_context}"
            )
        if validation_feedback is not None:
            prompt += (
                "\n\nThe previous response was invalid. Correct it without changing the task."
                f"\nValidation error: {validation_feedback}"
            )
            if previous_output:
                prompt += f"\nPrevious response:\n{previous_output[:2_000]}"
        return GenerationRequest(
            system_prompt=version.instructions,
            user_prompt=prompt,
            operation=f"agent:{version.agent_id}@{version.version}",
        )


class _LangGraphPromptState(TypedDict):
    mode: str
    system_prompt: str
    operation: str
    run_input: JsonObject
    output_schema: JsonObject
    definitions: list[JsonObject]
    previous_output: str | None
    validation_feedback: str | None
    tool_execution: ToolExecution | None
    action_execution: ActionExecution | None
    input_json: str
    definitions_json: str
    tool_context: str | None
    action_context: str | None
    user_prompt: str


def _serialize_graph_context(state: _LangGraphPromptState) -> dict[str, object]:
    return {
        "input_json": json.dumps(state["run_input"], separators=(",", ":"), ensure_ascii=False),
        "definitions_json": json.dumps(
            state["definitions"], separators=(",", ":"), ensure_ascii=False
        ),
        "tool_context": (
            None
            if state["tool_execution"] is None
            else json.dumps(
                state["tool_execution"].model_dump(mode="json"),
                separators=(",", ":"),
                ensure_ascii=False,
            )
        ),
        "action_context": (
            None
            if state["action_execution"] is None
            else json.dumps(
                state["action_execution"].model_dump(mode="json"),
                separators=(",", ":"),
                ensure_ascii=False,
            )
        ),
    }


def _compose_graph_prompt(state: _LangGraphPromptState) -> dict[str, str]:
    if state["mode"] == "action":
        prompt = (
            f"Input:\n{state['input_json']}\n\n"
            f"Available actions:\n{state['definitions_json']}\n\n"
            "Select exactly one available action needed to cause the requested controlled effect. "
            'Return exactly one JSON object shaped as {"action":{"action_type":"name",'
            '"payload":{...}}}. The payload must conform only to the selected input_schema; never '
            "copy output_schema fields into it. Do not include markdown or explanatory prose."
        )
        return {"user_prompt": prompt}

    if state["mode"] == "tool":
        prompt = (
            f"Input:\n{state['input_json']}\n\n"
            f"Available tools:\n{state['definitions_json']}\n\n"
            "Select exactly one available tool required to complete the task. Return exactly one "
            'JSON object shaped as {"tool_call":{"name":"tool_name","arguments":{...}}}. '
            "Do not include markdown or explanatory prose."
        )
        return {"user_prompt": prompt}

    schema = json.dumps(state["output_schema"], separators=(",", ":"), ensure_ascii=False)
    prompt = (
        f"Input:\n{state['input_json']}\n\n"
        f"Return exactly one JSON object conforming to this JSON Schema:\n{schema}\n\n"
        "Do not include markdown or explanatory prose."
    )
    if state["tool_context"] is not None:
        prompt += (
            "\n\nA platform-controlled tool returned this validated result. Use it when "
            f"producing the final answer:\n{state['tool_context']}"
        )
    if state["action_context"] is not None:
        prompt += (
            "\n\nA platform-controlled action produced this validated and verified result. "
            f"Use it when producing the final answer:\n{state['action_context']}"
        )
    if state["validation_feedback"] is not None:
        prompt += (
            "\n\nThe previous response was invalid. Correct it without changing the task."
            f"\nValidation error: {state['validation_feedback']}"
        )
        if state["previous_output"]:
            prompt += f"\nPrevious response:\n{state['previous_output'][:2_000]}"
    return {"user_prompt": prompt}


class LangGraphJsonHarness:
    """A graph-based harness that delegates platform authority to the existing runtime."""

    def __init__(self) -> None:
        workflow = StateGraph(_LangGraphPromptState)
        workflow.add_node("serialize_context", _serialize_graph_context)
        workflow.add_node("compose_prompt", _compose_graph_prompt)
        workflow.add_edge(START, "serialize_context")
        workflow.add_edge("serialize_context", "compose_prompt")
        workflow.add_edge("compose_prompt", END)
        self._graph = workflow.compile()

    def _build(
        self,
        *,
        mode: str,
        version: AgentVersion,
        run_input: JsonObject,
        definitions: list[JsonObject] | None = None,
        previous_output: str | None = None,
        validation_feedback: str | None = None,
        tool_execution: ToolExecution | None = None,
        action_execution: ActionExecution | None = None,
    ) -> GenerationRequest:
        state = self._graph.invoke(
            {
                "mode": mode,
                "system_prompt": version.instructions,
                "operation": f"{mode}:{version.agent_id}@{version.version}",
                "run_input": run_input,
                "output_schema": version.output_schema,
                "definitions": definitions or [],
                "previous_output": previous_output,
                "validation_feedback": validation_feedback,
                "tool_execution": tool_execution,
                "action_execution": action_execution,
                "input_json": "",
                "definitions_json": "",
                "tool_context": None,
                "action_context": None,
                "user_prompt": "",
            }
        )
        return GenerationRequest(
            system_prompt=state["system_prompt"],
            user_prompt=state["user_prompt"],
            operation=f"langgraph:{state['operation']}",
        )

    def build_action_request(
        self,
        version: AgentVersion,
        run_input: JsonObject,
        actions: list[ActionDefinition],
    ) -> GenerationRequest:
        return self._build(
            mode="action",
            version=version,
            run_input=run_input,
            definitions=[action.model_dump(mode="json") for action in actions],
        )

    def build_tool_request(
        self,
        version: AgentVersion,
        run_input: JsonObject,
        tools: list[ToolDefinition],
    ) -> GenerationRequest:
        return self._build(
            mode="tool",
            version=version,
            run_input=run_input,
            definitions=[tool.model_dump(mode="json") for tool in tools],
        )

    def build_request(
        self,
        version: AgentVersion,
        run_input: JsonObject,
        *,
        previous_output: str | None = None,
        validation_feedback: str | None = None,
        tool_execution: ToolExecution | None = None,
        action_execution: ActionExecution | None = None,
    ) -> GenerationRequest:
        return self._build(
            mode="agent",
            version=version,
            run_input=run_input,
            previous_output=previous_output,
            validation_feedback=validation_feedback,
            tool_execution=tool_execution,
            action_execution=action_execution,
        )


class HarnessRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, HarnessAdapter] = {
            "simple_json": SimpleJsonHarness(),
            "bounded_json": SimpleJsonHarness(),
            "langgraph_json": LangGraphJsonHarness(),
        }

    def get(self, name: str) -> HarnessAdapter:
        try:
            return self._adapters[name]
        except KeyError as error:
            raise HarnessNotFoundError(f"Unknown harness: {name}") from error
