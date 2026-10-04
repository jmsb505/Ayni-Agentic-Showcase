from __future__ import annotations

import hashlib
import logging
import re
from copy import deepcopy
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from typing import Any, Protocol
from uuid import UUID, uuid4

from pydantic import AwareDatetime, BaseModel, Field, HttpUrl, model_validator

from app.memory_retrieval import MemoryQuery

logger = logging.getLogger(__name__)

JsonObject = dict[str, Any]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RecordConflictError(RuntimeError):
    pass


class RecordNotFoundError(LookupError):
    pass




class KnowledgeSourceType(str, Enum):
    CANONICAL = "canonical"
    CLIENT_OPERATIONAL = "client_operational"
    EXTERNAL_RESEARCH = "external_research"


class MemoryStatus(str, Enum):
    CANDIDATE = "candidate"
    APPROVED = "approved"
    REJECTED = "rejected"


class KnowledgeCollectionCreate(BaseModel):
    deployment_id: str = Field(min_length=1, max_length=80)
    collection_key: str = Field(
        min_length=1,
        max_length=80,
        pattern=r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$",
    )
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2_000)


class KnowledgeCollection(KnowledgeCollectionCreate):
    collection_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=utc_now)


class KnowledgeDocumentCreate(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    content: str = Field(min_length=1, max_length=100_000)
    source_type: KnowledgeSourceType
    source_url: HttpUrl | None = None
    source_date: datetime | None = None


class KnowledgeDocument(KnowledgeDocumentCreate):
    document_id: UUID = Field(default_factory=uuid4)
    collection_id: UUID
    content_hash: str
    created_at: datetime = Field(default_factory=utc_now)


class KnowledgeCitation(BaseModel):
    document_id: UUID
    title: str
    source_url: HttpUrl | None
    source_date: datetime | None
    excerpt: str
    score: float = Field(ge=0)


class KnowledgeSearchRequest(BaseModel):
    deployment_id: str = Field(min_length=1, max_length=80)
    query: str = Field(min_length=2, max_length=500)
    collection_keys: list[str] = Field(default_factory=list, max_length=20)
    limit: int = Field(default=5, ge=1, le=12)


class EpisodicMemoryCreate(BaseModel):
    deployment_id: str = Field(min_length=1, max_length=80)
    agent_id: str | None = Field(default=None, max_length=128)
    run_id: UUID | None = None
    summary: str = Field(min_length=1, max_length=8_000)
    source_references: list[str] = Field(default_factory=list, max_length=20)


class EpisodicMemory(EpisodicMemoryCreate):
    episode_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=utc_now)


class SemanticMemoryCreate(BaseModel):
    deployment_id: str = Field(min_length=1, max_length=80)
    subject: str = Field(min_length=1, max_length=300)
    predicate: str = Field(min_length=1, max_length=160)
    value: JsonObject = Field(default_factory=dict)
    confidence: int = Field(ge=0, le=100)
    source_references: list[str] = Field(min_length=1, max_length=20)
    entity_id: str | None = Field(default=None, min_length=1, max_length=160)
    object_entity_id: str | None = Field(default=None, min_length=1, max_length=160)
    valid_from: AwareDatetime | None = None
    valid_until: AwareDatetime | None = None

    @model_validator(mode="after")
    def valid_interval(self):
        if self.valid_from and self.valid_until and self.valid_until <= self.valid_from:
            raise ValueError("valid_until must be after valid_from")
        if self.object_entity_id and not self.entity_id:
            raise ValueError("Relationships require a stable subject entity_id")
        return self


class SemanticMemory(SemanticMemoryCreate):
    memory_id: UUID = Field(default_factory=uuid4)
    status: MemoryStatus = MemoryStatus.CANDIDATE
    created_at: datetime = Field(default_factory=utc_now)
    approved_at: datetime | None = None
    superseded_at: AwareDatetime | None = None
    revision: int = Field(default=1, ge=1)


class ProceduralMemoryCreate(BaseModel):
    deployment_id: str = Field(min_length=1, max_length=80)
    agent_id: str | None = Field(default=None, max_length=128)
    title: str = Field(min_length=1, max_length=300)
    instruction: str = Field(min_length=1, max_length=8_000)
    supporting_evidence: list[str] = Field(min_length=1, max_length=20)


class ProceduralMemory(ProceduralMemoryCreate):
    procedure_id: UUID = Field(default_factory=uuid4)
    status: MemoryStatus = MemoryStatus.CANDIDATE
    created_at: datetime = Field(default_factory=utc_now)
    approved_at: datetime | None = None


class KnowledgeMemoryStore(Protocol):
    def retrieve_memory(self, request: MemoryQuery) -> JsonObject: ...

    def create_collection(self, request: KnowledgeCollectionCreate) -> KnowledgeCollection: ...

    def list_collections(self, deployment_id: str) -> list[KnowledgeCollection]: ...

    def list_documents(
        self, deployment_id: str, collection_keys: list[str]
    ) -> list[KnowledgeDocument]: ...

    def ingest_document(
        self, collection_id: UUID, request: KnowledgeDocumentCreate
    ) -> KnowledgeDocument: ...

    def search(self, request: KnowledgeSearchRequest) -> list[KnowledgeCitation]: ...

    def record_episode(self, request: EpisodicMemoryCreate) -> EpisodicMemory: ...

    def list_episodes(self, deployment_id: str, limit: int = 50) -> list[EpisodicMemory]: ...

    def propose_semantic(self, request: SemanticMemoryCreate) -> SemanticMemory: ...

    def approve_semantic(
        self,
        deployment_id: str,
        memory_id: UUID,
        expected_previous_id: UUID | None = None,
    ) -> SemanticMemory: ...

    def reject_semantic(self, deployment_id: str, memory_id: UUID) -> SemanticMemory: ...

    def list_semantic(
        self, deployment_id: str, status: MemoryStatus | None = None, limit: int = 50
    ) -> list[SemanticMemory]: ...

    def propose_procedural(self, request: ProceduralMemoryCreate) -> ProceduralMemory: ...

    def approve_procedural(self, deployment_id: str, procedure_id: UUID) -> ProceduralMemory: ...

    def reject_procedural(self, deployment_id: str, procedure_id: UUID) -> ProceduralMemory: ...

    def list_procedural(
        self, deployment_id: str, status: MemoryStatus | None = None, limit: int = 50
    ) -> list[ProceduralMemory]: ...


def approved_agent_memory_context(
    store: KnowledgeMemoryStore,
    deployment_id: str,
    agent_id: str,
    *,
    limit: int = 12,
    query: str = "",
    entity_ids: list[str] | None = None,
) -> JsonObject:
    """Return only explicitly approved, deployment-scoped memory for one agent.

    Memory is context, not an instruction override: callers must place this block below their
    authoritative system/deployment instructions. Agent-scoped procedures are only visible to
    their named agent; a null agent_id is intentionally shared within the deployment.
    """
    from app.memory_retrieval import MemoryQuery, retrieve_memory

    if callable(getattr(store, "retrieve_memory", None)):
        return retrieve_memory(
            store,
            MemoryQuery(
                deployment_id=deployment_id,
                agent_id=agent_id,
                query=query[:500],
                entity_ids=entity_ids or [],
                limit=min(max(limit, 1), 12),
            ),
        )
    # Keep lightweight application/test stores that predate memory listing compatible; the
    # concrete stores implement both methods and therefore always receive approved context.
    list_semantic = getattr(store, "list_semantic", None)
    list_procedural = getattr(store, "list_procedural", None)
    semantic = (
        list_semantic(deployment_id, MemoryStatus.APPROVED, limit=limit)
        if callable(list_semantic)
        else []
    )
    procedures = (
        list_procedural(deployment_id, MemoryStatus.APPROVED, limit=limit)
        if callable(list_procedural)
        else []
    )
    procedural = [item for item in procedures if item.agent_id is None or item.agent_id == agent_id]
    return {
        "semantic": [
            {
                "memory_id": str(item.memory_id),
                "subject": item.subject,
                "predicate": item.predicate,
                "value": item.value,
                "confidence": item.confidence,
                "source_references": item.source_references,
            }
            for item in semantic
        ],
        "procedural": [
            {
                "procedure_id": str(item.procedure_id),
                "title": item.title,
                "instruction": item.instruction,
                "supporting_evidence": item.supporting_evidence,
            }
            for item in procedural
        ],
    }


def _chunks(content: str, size: int = 1_200) -> list[str]:
    words = content.split()
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for word in words:
        if current and current_length + len(word) + 1 > size:
            chunks.append(" ".join(current))
            current, current_length = [], 0
        current.append(word)
        current_length += len(word) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class InMemoryKnowledgeMemoryStore:
    def __init__(self) -> None:
        self._memory_lock = RLock()
        self._collections: dict[UUID, KnowledgeCollection] = {}
        self._documents: dict[UUID, KnowledgeDocument] = {}
        self._document_chunks: dict[UUID, list[str]] = {}
        self._episodes: dict[UUID, EpisodicMemory] = {}
        self._semantic: dict[UUID, SemanticMemory] = {}
        self._procedural: dict[UUID, ProceduralMemory] = {}

    def retrieve_memory(self, request):
        from app.memory_retrieval import in_memory_retrieve

        with self._memory_lock:
            return in_memory_retrieve(self, request)

    def create_collection(self, request: KnowledgeCollectionCreate) -> KnowledgeCollection:
        if any(
            value.deployment_id == request.deployment_id
            and value.collection_key == request.collection_key
            for value in self._collections.values()
        ):
            raise RecordConflictError("Knowledge collection key already exists for deployment.")
        collection = KnowledgeCollection(**request.model_dump())
        self._collections[collection.collection_id] = deepcopy(collection)
        return collection

    def list_collections(self, deployment_id: str) -> list[KnowledgeCollection]:
        return [
            deepcopy(value)
            for value in self._collections.values()
            if value.deployment_id == deployment_id
        ]

    def list_documents(
        self, deployment_id: str, collection_keys: list[str]
    ) -> list[KnowledgeDocument]:
        allowed_ids = {
            collection.collection_id
            for collection in self._collections.values()
            if collection.deployment_id == deployment_id
            and (not collection_keys or collection.collection_key in collection_keys)
        }
        return [
            deepcopy(document)
            for document in self._documents.values()
            if document.collection_id in allowed_ids
        ]

    def ingest_document(
        self, collection_id: UUID, request: KnowledgeDocumentCreate
    ) -> KnowledgeDocument:
        if collection_id not in self._collections:
            raise RecordNotFoundError(f"Knowledge collection not found: {collection_id}")
        document = KnowledgeDocument(
            collection_id=collection_id,
            content_hash=_content_hash(request.content),
            **request.model_dump(),
        )
        self._documents[document.document_id] = deepcopy(document)
        self._document_chunks[document.document_id] = _chunks(document.content)
        return document

    def search(self, request: KnowledgeSearchRequest) -> list[KnowledgeCitation]:
        query_terms = set(re.findall(r"\w+", request.query.casefold()))
        matches: list[KnowledgeCitation] = []
        for document_id, document in self._documents.items():
            collection = self._collections[document.collection_id]
            if collection.deployment_id != request.deployment_id:
                continue
            if request.collection_keys and collection.collection_key not in request.collection_keys:
                continue
            for chunk in self._document_chunks[document_id]:
                score = sum(term in chunk.casefold() for term in query_terms)
                if score:
                    matches.append(
                        KnowledgeCitation(
                            document_id=document_id,
                            title=document.title,
                            source_url=document.source_url,
                            source_date=document.source_date,
                            excerpt=chunk,
                            score=float(score),
                        )
                    )
        return sorted(matches, key=lambda item: item.score, reverse=True)[: request.limit]

    def record_episode(self, request: EpisodicMemoryCreate) -> EpisodicMemory:
        memory = EpisodicMemory(**request.model_dump())
        self._episodes[memory.episode_id] = deepcopy(memory)
        return memory

    def list_episodes(self, deployment_id: str, limit: int = 50) -> list[EpisodicMemory]:
        return sorted(
            [
                deepcopy(value)
                for value in self._episodes.values()
                if value.deployment_id == deployment_id
            ],
            key=lambda value: value.created_at,
            reverse=True,
        )[:limit]

    def propose_semantic(self, request: SemanticMemoryCreate) -> SemanticMemory:
        with self._memory_lock:
            for existing in self._semantic.values():
                if (
                    existing.status is MemoryStatus.CANDIDATE
                    and SemanticMemoryCreate(**existing.model_dump()) == request
                ):
                    return deepcopy(existing)
            memory = SemanticMemory(**request.model_dump())
            self._semantic[memory.memory_id] = deepcopy(memory)
            return memory

    def approve_semantic(
        self,
        deployment_id: str,
        memory_id: UUID,
        expected_previous_id: UUID | None = None,
    ) -> SemanticMemory:
        with self._memory_lock:
            try:
                memory = self._semantic[memory_id]
            except KeyError as error:
                raise RecordNotFoundError(f"Semantic memory not found: {memory_id}") from error
            if memory.deployment_id != deployment_id:
                raise RecordNotFoundError(f"Semantic memory not found: {memory_id}")
            return self._approve_semantic_locked(memory_id, expected_previous_id)

    def _approve_semantic_locked(self, memory_id, expected_previous_id):
        try:
            memory = self._semantic[memory_id]
        except KeyError as error:
            raise RecordNotFoundError(f"Semantic memory not found: {memory_id}") from error
        if memory.status is not MemoryStatus.CANDIDATE:
            raise RecordConflictError("Only a candidate can be approved.")
        current = [
            x
            for x in self._semantic.values()
            if x.deployment_id == memory.deployment_id
            and (x.entity_id or x.subject) == (memory.entity_id or memory.subject)
            and x.predicate == memory.predicate
            and x.status is MemoryStatus.APPROVED
            and x.superseded_at is None
        ]
        expected = [expected_previous_id] if expected_previous_id else []
        if [x.memory_id for x in current] != expected:
            raise RecordConflictError(
                "Current fact changed; review it before approving replacement."
            )
        now = utc_now()
        for previous in current:
            self._semantic[previous.memory_id] = previous.model_copy(update={"superseded_at": now})
        approved = memory.model_copy(
            update={
                "status": MemoryStatus.APPROVED,
                "approved_at": now,
                "revision": current[0].revision + 1 if current else 1,
            }
        )
        self._semantic[memory_id] = approved
        return deepcopy(approved)

    def list_semantic(
        self, deployment_id: str, status: MemoryStatus | None = None, limit: int = 50
    ) -> list[SemanticMemory]:
        return sorted(
            [
                deepcopy(value)
                for value in self._semantic.values()
                if value.deployment_id == deployment_id
                and (status is None or value.status is status)
            ],
            key=lambda value: value.created_at,
            reverse=True,
        )[:limit]

    def propose_procedural(self, request: ProceduralMemoryCreate) -> ProceduralMemory:
        memory = ProceduralMemory(**request.model_dump())
        self._procedural[memory.procedure_id] = deepcopy(memory)
        return memory

    def approve_procedural(self, deployment_id: str, procedure_id: UUID) -> ProceduralMemory:
        with self._memory_lock:
            try:
                memory = self._procedural[procedure_id]
            except KeyError as error:
                raise RecordNotFoundError(f"Procedural memory not found: {procedure_id}") from error
            if memory.deployment_id != deployment_id:
                raise RecordNotFoundError(f"Procedural memory not found: {procedure_id}")
            if memory.status is not MemoryStatus.CANDIDATE:
                raise RecordConflictError("Only a candidate can be approved.")
            approved = memory.model_copy(
                update={"status": MemoryStatus.APPROVED, "approved_at": utc_now()}
            )
            self._procedural[procedure_id] = approved
            return deepcopy(approved)

    def reject_semantic(self, deployment_id: str, memory_id: UUID) -> SemanticMemory:
        with self._memory_lock:
            try:
                memory = self._semantic[memory_id]
            except KeyError as error:
                raise RecordNotFoundError(f"Semantic memory not found: {memory_id}") from error
            if memory.deployment_id != deployment_id:
                raise RecordNotFoundError(f"Semantic memory not found: {memory_id}")
            if memory.status is not MemoryStatus.CANDIDATE:
                raise RecordConflictError("Only a candidate can be rejected.")
            rejected = memory.model_copy(update={"status": MemoryStatus.REJECTED})
            self._semantic[memory_id] = rejected
            return deepcopy(rejected)

    def reject_procedural(self, deployment_id: str, procedure_id: UUID) -> ProceduralMemory:
        with self._memory_lock:
            try:
                memory = self._procedural[procedure_id]
            except KeyError as error:
                raise RecordNotFoundError(f"Procedural memory not found: {procedure_id}") from error
            if memory.deployment_id != deployment_id:
                raise RecordNotFoundError(f"Procedural memory not found: {procedure_id}")
            if memory.status is not MemoryStatus.CANDIDATE:
                raise RecordConflictError("Only a candidate can be rejected.")
            rejected = memory.model_copy(update={"status": MemoryStatus.REJECTED})
            self._procedural[procedure_id] = rejected
            return deepcopy(rejected)

    def list_procedural(
        self, deployment_id: str, status: MemoryStatus | None = None, limit: int = 50
    ) -> list[ProceduralMemory]:
        return sorted(
            [
                deepcopy(value)
                for value in self._procedural.values()
                if value.deployment_id == deployment_id
                and (status is None or value.status is status)
            ],
            key=lambda value: value.created_at,
            reverse=True,
        )[:limit]


