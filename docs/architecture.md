# Ayni agent architecture

## Operating model

Ayni agents are specialized capabilities inside a controlled workflow. A stage receives a typed request, uses its pinned harness and provider path, produces a structured result, and returns control to workflow code. Workflow services decide which stage is eligible to run next. The language model does not create arbitrary agents or choose an unrestricted sequence of actions.

There is no general-purpose agent swarm in this design. The LangGraph harness is a small graph for serializing context and composing a generation prompt. It does not replace the workflow, Run lifecycle, authorization, or evidence controls.

## Ownership boundaries

| Capability | Owns | Kept outside this repository |
| --- | --- | --- |
| Agent workflow | Stage-specific reasoning request, typed result, evidence checks, and declared decision points | Application record storage and the shared execution service |
| Harness | How an agent request is assembled for generation | Run identity, policy, external effects, and persistence |
| Model provider | Provider-specific generation call and usage metadata | Workflow transitions, approval, and business records |
| Judge provider | Typed semantic assessment for a declared question | Threshold policy, permissions, approvals, and mutation |
| Memory | Scoped retrieval and proposed, approved, or rejected memory state | Authority to override instructions or take actions |
| Connector | Versioned communication envelope and normalized outcomes | The business meaning and authorization policy of each port |
| Capsule | Private implementation behind declared typed ports | A package marketplace or arbitrary runtime installation |
| Ayni Core host | Run lifecycle, actor and tenant enforcement, budgets, evidence, approvals, recovery, and evaluation | Not copied into this repository |
| Application | Business records and workflow transactions | CRM product, UI, API, and data are not copied |

The application composition root may bind implementations to ports. Business logic does not cross capsule ownership by importing another capsule's internals.

## Capsule composition

Each capsule declares a stable identity and version, kind, provided and required ports, compatible versions, configuration shape, capabilities, permissions, and readiness. A statically installed registry validates the manifests and a deployment selects exact versions. An agent version pins its required harness and capability contracts; a Run release records the resolved composition so a later provider change cannot rewrite the meaning of old work.

Capsules are logical modules. The design keeps them in a modular monolith by default. An in-process adapter and a remote adapter should obey the same connector semantics. A separate service is justified by operational evidence such as isolation, scaling, availability, or ownership needs. The manifest examples here document the contract; the registry, deployment service, and Core application are intentionally absent.

## Connector message

A connector request identifies the exact source and target capsule versions and exact port contract version. It carries tenant and deployment scope, initiating actor context, correlation and Run references, a stable operation identity, an effect class, and the port's typed payload.

The receiver authenticates the caller and enforces tenant membership and permission. The actor and tenant values inside a message are context; they are not credentials. The receiver validates the payload against the port contract. It returns one of four outcomes:

- accepted asynchronous work with a stable operation reference;
- succeeded work with a typed result;
- failed work with a stable error and retry policy;
- uncertain work with a reconciliation reference when an effect may have happened but is unproven.

For consequential effects, success also carries a provider receipt. A connector acknowledgement alone does not prove delivery or mutation. Uncertain outcomes are reconciled before retry so an operation is not blindly duplicated.

## Interchangeable execution pieces

### Harness

The harness contract accepts the agent version, input, and any validated tool or action result, then returns a generation request. Simple JSON and LangGraph JSON implement the same request-building interface. The output schema and harness identity are part of the agent version. A new harness can be bound behind the same typed port if it preserves the contract and passes the same workflow evaluation.

### Model

The model router chooses among deployment-configured model providers. The provider contract accepts system and user prompts plus operation metadata and returns content and usage details. The source snapshot contains local and hosted provider adapters. The router and deployment profile are host services and were not copied, so this repository documents provider interchange without claiming a local provider switch can be executed here.

### Judge

Semantic decisions use the provider-neutral Judge contract by default. A Judge request carries bounded state plus explicitly named Choice, Score, or Noul questions. Provider results have the matching typed answer and usage/latency metadata. Workflow code applies a versioned threshold or routes uncertainty to a person. A provider swap must retain the same question/result meaning and pass the same offline evaluation.

### Search, knowledge, and external tools

Research tools return typed source observations. Deterministic validation binds a generated claim to known source identifiers and rejects missing, unknown, or wrong-scope citations. A search provider or knowledge source can be replaced behind its own port. Search snippets are candidate evidence, not truth, and approved memory is not public research.

## One workflow pass

1. An operator or application supplies a bounded request and approved context.
2. The host creates a Run using an immutable agent version and snapshots its capsule/model composition, tenant, and budget.
3. Workflow code invokes a declared stage. It decides the order; the model only produces the stage's requested output.
4. The selected harness composes a structured request and calls the model through its versioned port.
5. If the stage needs search or knowledge, it calls the declared capability port and keeps source IDs and citation metadata with the result.
6. Deterministic code validates schemas, citations, eligibility, and policy. Semantic quality or fit checks use the Judge when declared; their threshold and fallback remain workflow policy.
7. A human reviews the profile, retained lead, proposal, or consequential action when the workflow requires it.
8. The host records typed events, model/Judge use, results, errors, and effect receipts. Memory suggestions stay candidates until reviewed.

The trace lets an operator reconstruct which input, evidence, exact versions, decisions, and controlled effects were involved. This repository omits the host trace service and does not contain a live Run inspector.

## Decision ownership

| Decision | Method | Example boundary |
| --- | --- | --- |
| Schema, citation membership, hard eligibility, authorization, deduplication, idempotency | Deterministic | Reject invalid output before a workflow transition |
| Contextual evidence support, readiness, fit, quality, or ranking | Judge by default | Typed questions and criteria; workflow applies the declared threshold |
| High-impact approval, conflicting evidence, or unresolved uncertainty | Human | Approve a profile or lead, review a proposal, authorize an effect |

Generation can propose an answer; it cannot grant permission, approve a record, or prove an external effect. The host retains authority for Runs, budgets, evidence, and recovery.

## Change rule

Adding a provider or harness means adding a manifest and a typed port binding, selecting an exact compatible version, and preserving that resolution in execution evidence. An incompatible payload change needs a new port contract version. Keep private implementation behind the port and compare replacements with the existing workflow evaluation before promoting them.
