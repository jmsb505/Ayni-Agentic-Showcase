# Ayni Agentic System

An interview-focused view of the agent workflows and architecture built for Ayni. This repository is a separate project beside the private source checkout. It contains no CRM product or Ayni Core service.

## What this repository contains

- Custom workflow stages for market research, lead discovery, lead research, proposal drafting, and controlled outreach.
- Structured-generation harnesses and the model-provider interface used by those stages.
- The semantic, episodic, and procedural memory model with bounded retrieval.
- The typed connector envelope, selected execution and Judge contracts, and capsule manifest examples.
- Guides to ownership, interchangeability, review boundaries, and the source extraction limits.

The Python modules are a sanitized source snapshot. They are useful for code review and an interview walkthrough, but they do not form an installable application. Their original imports refer to host services and business types that are intentionally absent here. The [isolation audit](docs/isolation-audit.md) lists the exact gaps.

## Is this isolation viable?

Yes, as a focused interview and code-showcase repository. It is not yet a standalone agent runtime. The actual workflow services depend on Ayni Core for Run identity and evidence, bounded execution, model selection, capsule resolution, connector dispatch, and recovery. Some stages also use application-owned organization and lead types. Those dependencies are described, not copied. Removing them without replacing their typed interfaces would produce a misleading or nonfunctional extraction.

The next step toward a runnable project is to define neutral workflow input, evidence, and result contracts, then adapt the existing stages to injected ports. A local fixture host could run the stages without a CRM application or production platform service.

## How the system works

The workflow is a sequence of bounded capabilities. Application code decides which stage runs next; an LLM does not invent the workflow or gain authority from its output.

    Human task and approved context
                |
                v
    Versioned agent capability
                |
                v
    Workflow stage selects its declared harness
                |
                v
    Harness builds a structured generation request
                |
                v
    Connector calls the selected model capsule and port
                |
                v
    Deterministic validation and, where required, Judge assessment
                |
         +------+------+
         |             |
         v             v
    Next bounded   Human review or
    workflow step  controlled action

The private host records Runs, versions, events, usage, errors, approvals, and recovery evidence. That operational service is not part of this repository. See the [architecture guide](docs/architecture.md).

## Harnesses and model providers

An agent version names a harness and an output contract. The simple JSON harness serializes the request and schema into a constrained prompt. The LangGraph JSON harness uses a small graph to prepare the prompt. It is a prompt-preparation graph, not a multi-agent supervisor or swarm. Bounded JSON shares structured generation while the host runtime applies bounded validation and retry rules.

The provider interface separates generation requests from model implementations. The source checkout includes local Qwen, OpenAI-compatible chat, and first-party OpenAI Responses adapters. Deployment configuration chooses a provider and model profile, and the execution release records the resolved choice. Swapping a provider does not transfer workflow policy or Run authority to the model.

## Capsules, connectors, and the Judge

A capsule is a versioned capability module with declared ports, dependencies, configuration, and permissions. An agent composition pins the capsule and port versions it needs. Deployments resolve those references to exact installed versions. This is a modular-monolith design: capsules may run in one process or behind a remote transport, and the boundary does not require a microservice split.

The connector envelope standardizes capsule identity, port version, tenant and actor context, correlation, operation identity, effect class, and outcome. Each port still owns its typed payload and policy. Actor fields are audit context, not proof of permission. A consequential success needs an effect receipt; an uncertain result must be reconciled before retry.

The Judge is a provider-neutral semantic assessment port, not another workflow agent. Workflow code applies the declared threshold. Deterministic logic handles exact checks, authorization, eligibility, and idempotency. A person handles high-impact choices and unresolved uncertainty. A Judge result cannot approve a record or authorize an external effect by itself.

Selected schemas, contracts, fixtures, and manifests are under [contracts](contracts/) and [capsules](capsules/). The registry, deployment service, connector dispatch, and Core application are not included.

## Three memory tiers

- **Semantic:** reviewed facts or assertions with sources, validity, deployment scope, and approval state.
- **Episodic:** short summaries linked to prior agent Runs. They help recall prior work but do not replace Run event history or become trusted facts automatically.
- **Procedural:** evidence-backed instructions or playbooks. A human must approve them before retrieval; agent-specific procedures stay scoped to that agent.

Knowledge collections hold source material for retrieval and citation. They are not a fourth memory tier. Run history and evidence are the operational record, not agent memory. Retrieved memory is context; it does not override instructions, count as public evidence, or grant authority. See the [memory guide](docs/memory-model.md).

## Agent stages

The included reference path is market research, lead discovery, lead research, and proposal drafting. Controlled outreach is a separate effect boundary. Each stage has typed outputs and its own deterministic, Judge, and human decision points. The model proposes content; the workflow validates it and controls transitions. The copied capability catalog also defines a read-only workspace assistant and a legacy CRM company-description capability. Their CRM-backed context and product adapters are excluded.

The [agent inventory](docs/agent-inventory.md) shows stage purpose and review boundaries. CRM-record enrichment and CRM product capabilities are excluded because they belong to the removed business application.

## Repository map

- [Architecture](docs/architecture.md) — ownership, capsules, connectors, provider interchange, decisions, and one workflow pass.
- [Agent inventory](docs/agent-inventory.md) — included workflow stages and their controls.
- [Memory model](docs/memory-model.md) — the three tiers, retrieval scope, and trust boundaries.
- [Isolation audit](docs/isolation-audit.md) — copied and excluded source, missing host interfaces, and the path to a runnable extraction.
- [Source snapshot](source/app/) — selected workflow, harness, provider, memory, Judge, and connector implementation files.
- [Contracts and manifests](contracts/) — selected generic contracts and fixtures, plus agent-facing capsule examples.

## Provenance and public-sharing status

This snapshot was curated from the Ayni working tree on 2026-10-04. It includes source edits that were uncommitted in the private checkout. It does not establish that those edits are hosted, enabled in a deployment, or accepted by an end-to-end quality evaluation. Organization and geography literals in copied source were generalized; the workflow structure and control logic were preserved except where those literal substitutions changed wording.

No CRM records, credentials, environment files, model weights, database content, private run traces, or the upstream CRM agent application are included. This repository has no license file. Add one if you intend to grant reuse rights.
