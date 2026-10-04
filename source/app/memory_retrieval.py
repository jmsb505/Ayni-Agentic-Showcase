"""Bounded, evidence-preserving retrieval. Stored assertions never grant authority."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from copy import deepcopy

from pydantic import BaseModel, Field

from app.domain import utc_now
from app.execution_controls import active_budget, stable_hash


class MemoryQuery(BaseModel):
    deployment_id: str = Field(min_length=1, max_length=80)
    agent_id: str = Field(min_length=1, max_length=128, pattern=r"^[a-z][a-z0-9._-]*$")
    query: str = Field(default="", max_length=500)
    entity_ids: list[str] = Field(default_factory=list, max_length=8)
    predicate: str | None = Field(default=None, max_length=160)
    limit: int = Field(default=6, ge=1, le=12)
    max_characters: int = Field(default=8_000, ge=512, le=24_000)
    relationship_hops: int = Field(default=0, ge=0, le=2)
    include_episodes: bool = False
    model_config = {"extra": "forbid"}


def retrieve_memory(store, request: MemoryQuery) -> dict:
    budget = active_budget.get()
    if budget:
        budget.check()
        request = request.model_copy(
            update={"max_characters": min(request.max_characters, budget.limits.memory_characters)}
        )
    key = stable_hash(request.model_dump())
    if budget and key in budget.memory_cache:
        return deepcopy(budget.memory_cache[key])
    raw = store.retrieve_memory(request)
    result = pack_memory(raw, request)
    if budget:
        # Bounded by tool/run budgets; additionally cap both maps explicitly.
        if len(budget.memory_cache) < 32:
            budget.memory_cache[key] = deepcopy(result)
        if len(budget.memory_evidence) < 32:
            budget.memory_evidence[key] = result["retrieval"]
    return result


def pack_memory(raw: dict, request: MemoryQuery) -> dict:
    result = {
        "semantic": [],
        "procedural": [],
        "episodes": [],
        "relationships": [],
        "conflicts": raw.get("conflicts", [])[:12],
    }
    metadata = {
        "strategy": "exact" if request.entity_ids else "lexical_recency",
        "deployment_id": request.deployment_id,
        "agent_id": request.agent_id,
        "memory_ids": [],
        "truncated": bool(raw.get("truncated", False)),
        "as_of": utc_now().isoformat(),
        "query_sha256": stable_hash(request.query),
    }
    result["retrieval"] = metadata
    # Conflicts are never silently resolved. Oversized records are omitted whole.
    for section in ("semantic", "procedural", "relationships", "episodes"):
        for item in raw.get(section, [])[: request.limit]:
            record = dict(item)
            record_id = str(
                record.get("memory_id") or record.get("procedure_id") or record.get("episode_id")
            )
            result[section].append(record)
            metadata["memory_ids"].append(record_id)
            if len(json.dumps(result, ensure_ascii=False, default=str)) > request.max_characters:
                result[section].pop()
                metadata["memory_ids"].pop()
                metadata["truncated"] = True
    if len(json.dumps(result, ensure_ascii=False, default=str)) > request.max_characters:
        # Keep an explicit conflict signal even when its detail exceeds the context allowance.
        result["conflicts"] = [{"unresolved": True}] if result["conflicts"] else []
        metadata["truncated"] = True
    return result


def in_memory_retrieve(store, request: MemoryQuery) -> dict:
    now = utc_now()
    terms = set(re.findall(r"\w+", request.query.casefold()))

    def visible(item):
        return item.deployment_id == request.deployment_id and getattr(item, "agent_id", None) in (
            None,
            request.agent_id,
        )

    def current(item):
        return (
            item.status.value == "approved"
            and item.superseded_at is None
            and (item.valid_from is None or item.valid_from <= now)
            and (item.valid_until is None or item.valid_until > now)
        )

    def relevance(item):
        content = json.dumps(item.model_dump(mode="json"), ensure_ascii=False).casefold()
        return sum(term in content for term in terms)

    facts = [x for x in store._semantic.values() if visible(x) and current(x)]
    entities = set(request.entity_ids)
    edges = []
    frontier = set(entities)
    for _ in range(request.relationship_hops):
        next_frontier = set()
        for entity in sorted(frontier):
            adjacent = sorted(
                (x for x in facts if x.entity_id == entity and x.object_entity_id),
                key=lambda x: str(x.memory_id),
            )[:8]
            for edge in adjacent:
                if edge.memory_id not in {x.memory_id for x in edges} and len(edges) < 32:
                    edges.append(edge)
                    next_frontier.add(edge.object_entity_id)
        frontier = next_frontier - entities
        entities.update(next_frontier)
    selected = [
        x
        for x in facts
        if (not request.entity_ids or x.entity_id in entities)
        and (request.predicate is None or x.predicate == request.predicate)
        and (request.entity_ids or not terms or relevance(x))
    ]
    selected.sort(
        key=lambda x: (x.entity_id in request.entity_ids, relevance(x), x.created_at), reverse=True
    )
    groups = defaultdict(list)
    for fact in facts:
        groups[(fact.entity_id or fact.subject, fact.predicate)].append(fact)
    conflicts = [
        {"subject": key[0], "predicate": key[1], "unresolved": True}
        for key, items in groups.items()
        if len({stable_hash(x.value) for x in items}) > 1
        and any(x in selected[: request.limit] for x in items)
    ]
    procedures = [
        x
        for x in store._procedural.values()
        if visible(x) and x.status.value == "approved" and (not terms or relevance(x))
    ]
    procedures.sort(key=lambda x: (relevance(x), x.created_at), reverse=True)
    episodes = [x for x in store._episodes.values() if visible(x) and relevance(x)]
    episodes.sort(key=lambda x: (relevance(x), x.created_at), reverse=True)
    sections = {
        "semantic": selected,
        "procedural": procedures,
        "episodes": episodes if request.include_episodes else [],
        "relationships": edges,
    }
    return {
        **{
            key: [x.model_dump(mode="json") for x in values[: request.limit]]
            for key, values in sections.items()
        },
        "conflicts": conflicts,
        "truncated": any(len(x) > request.limit for x in sections.values()),
    }
