# Ayni Agentic System

Ayni is a modular system for running bounded, evidence-aware agent workflows. It turns a human-defined objective and approved context into structured work, checks the result, and routes it to the next stage, a person, or an explicitly approved action.

The system combines focused agent capabilities with a controlled workflow host. The host owns run identity and lifecycle, version selection, authorization, budgets, evidence, and recovery. Each capability owns a specific reasoning task and returns a typed result. A model can draft or assess that result; workflow code controls what happens next.

## What the system does

The main workflow moves from research to reviewed action:

1. Market research builds an operating-context profile and evidence-linked market findings.
2. Lead discovery searches within a bounded scope, screens candidates, checks duplicates, and queues results for review.
3. Lead research builds a cited profile of a retained candidate, including unknowns and risks.
4. Proposal drafting creates a proposal and related outreach text from approved context.
5. Controlled outreach sends only the exact recipient and content that were approved.

A read-only workspace assistant can answer questions from supplied workspace context and run evidence. It does not advance workflows or mutate records.

Workflow stages are application-orchestrated. The selected agent handles its declared task and returns control to the workflow. LangGraph is used in one harness to compose a prompt; it does not choose an unrestricted chain of agents or actions.

## How a run works

An operator or application supplies the task and approved context. The host starts a run with a versioned capability, its pinned capsule and port contracts, and an execution budget.

    Human goal + approved context
                 ↓
    Run orchestration: identity, versions, budget, evidence
                 ↓
    Versioned workflow capability
          ┌──────┴─────────┐
          ↓                ↓
    Selected harness   Typed connector calls
          ↓                ↓
    Model provider     Search, knowledge, memory,
          ↓            Judge, or action capability
    Structured result        │
          └────────┬─────────┘
                   ↓
    Deterministic checks and any required Judge or human review
                   ↓
    Next workflow stage, review queue, or approved action
                   ↓
    Run trace and outcome evidence

Each run records the versions and decisions that produced its result. For consequential effects, the receiver returns an effect receipt. An uncertain result is reconciled before a retry.

## Capsules and interchangeable components

A capsule is a versioned capability module with declared ports, dependencies, configuration, capabilities, and permissions. An agent composition pins the capsule and port contract versions it uses. The host resolves those pins and records the exact composition with each run.

Capsules are logical boundaries. The default deployment is a modular monolith: modules can run in one process, while a remote adapter can use the same connector contract when deployment needs it. The architecture does not require a separate service for every capability.

The shared connector envelope carries capsule and port identity, version, tenant and actor context, correlation and operation identity, effect class, and outcome. Each port defines its own typed payload and policy. The receiver enforces authorization; identity fields in a message provide context, not permission. Consequential success includes a provider receipt, and uncertain outcomes require reconciliation before retry.

The design keeps several parts replaceable behind stable contracts:

- **Harness:** bounded JSON and simple JSON prepare structured generation requests. The LangGraph JSON harness uses a small graph to serialize context and compose the prompt.
- **Model provider:** a common interface accepts prompts and operation metadata, then returns generated content and usage. Implementations include local Qwen, OpenAI-compatible chat, and OpenAI Responses. A deployment can select a local model or a hosted provider without moving workflow policy into the model.
- **Judge:** a provider-neutral port returns typed semantic assessments, as a choice among labels, a score against ordered criteria, or a probability estimate for one condition. Workflow policy applies thresholds and decides when uncertainty needs a person.
- **Tools and knowledge:** search, source collections, memory, and external actions have typed ports, so an implementation can change while its contract remains stable.

A replacement keeps the same port meaning, declares its capsule version, and passes the workflow evaluation. A provider or harness change does not silently change a past run’s recorded composition.

## Three-tier memory

The memory tiers answer three different questions: what is known, what happened, and how a task should be done.

| Tier | Stores | Why it is useful |
| --- | --- | --- |
| **Semantic** | Reviewed, source-linked facts with scope, validity, confidence, and revision | Carries current knowledge across runs without repeating discovery. Retrieval can filter out unapproved or outdated facts. |
| **Episodic** | Short summaries linked to the run, capability, time, and source references | Helps a capability recall prior attempts and outcomes. The original run trace remains the record of what happened. |
| **Procedural** | Evidence-backed instructions and playbooks, optionally scoped to one capability | Preserves approved methods so later runs can apply them consistently. |

Separating these tiers keeps facts, experience, and instructions from being treated as the same kind of information. Retrieval is scoped and bounded so a prompt receives relevant context instead of an unrestricted history dump. Semantic facts and procedures require review before use; episodes can be requested explicitly and remain linked to their source run.

Knowledge collections are source material for retrieval and citations, not a fourth memory tier. Run history is operational evidence, not memory. Retrieved memory supplies context; it does not prove a public claim or grant authority.

## A governed improvement loop

Ayni improves from run outcomes through evaluation and reviewed updates:

1. The host records inputs, evidence, resolved versions, decisions, usage, errors, and action receipts.
2. Deterministic checks catch contract, citation, eligibility, and policy failures. Where a semantic assessment is needed, the Judge evaluates the declared question against bounded evidence.
3. Run outcomes can inform episodic summaries and candidates for semantic facts or procedural memory. Owners review memory candidates before approval.
4. Evaluations compare workflow results against defined quality criteria and regression cases.
5. Improvements to instructions, thresholds, harnesses, provider bindings, and procedures are introduced as versioned changes and evaluated before promotion.

This loop lets the system reuse reviewed knowledge and improve later versions based on measured outcomes. Changes remain traceable to their runs and versions. Generated text does not change permissions or workflow code as a side effect.

## Decision and action boundaries

Deterministic code handles exact rules such as schema validity, citation membership, eligibility, authorization, deduplication, and idempotency. The Judge assesses questions that require semantic judgment, such as evidence support, fit, readiness, or quality. People review high-impact choices and authorize consequential actions when the workflow requires it.

A Judge result is evidence for a workflow decision; it is not permission to approve a record or perform an external effect. The workflow applies its declared policy and records the decision. External effects use a separate action boundary with approval, receipt verification, and reconciliation for uncertain outcomes.

## Project layout and scope

This repository contains workflow capabilities, harness and provider implementations, memory and Judge logic, connector contracts, capsule manifests, and architecture guides. It is an architectural reference rather than a one-command standalone service. Execution relies on host services for run lifecycle, capsule resolution, persistence, authorization, budgets, evidence, recovery, and evaluation.

- [Architecture](docs/architecture.md) — ownership, composition, connectors, providers, decisions, and a workflow pass.
- [Agent inventory](docs/agent-inventory.md) — capability purposes and review boundaries.
- [Memory model](docs/memory-model.md) — memory tiers, retrieval scope, and trust rules.
- [Reference modules](source/app/) — workflow, harness, provider, memory, Judge, and connector code.
- [Contracts and capsule manifests](contracts/) — schemas, port descriptions, fixtures, and versioned capability examples.
