# Isolation audit

## What was copied

This repository contains selected source from the private Ayni backend, plus generic contracts and agent-facing manifests:

- Workflow source: capability catalog, Company Brain request handling, market research, lead discovery, lead research, proposal drafting, and controlled delivery.
- Execution source: structured-generation harnesses and model-provider interface/adapters.
- Memory source: memory schemas and the in-memory store excerpt, plus bounded retrieval. The database-backed adapter was removed.
- Decision source: Judge adapter and shared Judge decision policy.
- Connector source: generic typed request/response envelope models.
- Contract files: connector and capsule schemas, execution/Judge ports, and neutral Judge and memory-review fixtures.
- Manifest examples: harness, model provider/router, Judge, and memory capsule declarations.

The selected Python files retain the source module layout where useful. Organization and target geography literals were generalized in the copies. The memory copy stops before the database-backed store and supplies local timestamp and error types. No edits were made to the private source checkout's implementation files for this extraction.

## What was deliberately left out

- The CRM web application, API, database schema, customer/company records, onboarding and workspace flows, CRM-specific UI, and CRM agent server.
- The Ayni Core API entry point and its Run lifecycle, authorization, persistence, queue, deployment activation, recovery, inspection, and evaluation services.
- CRM-specific Company and Prospect stores, database adapters, external research configuration, search credentials, and outbound delivery providers.
- Local deployment configuration, environment files, credentials, provider keys, model weights, logs, screenshots, and generated run traces.
- The imported upstream CRM repository and its source code.

## Why the source snapshot is not runnable

Included workflow modules import host services and business types that are absent here. Copying their entire dependency tree would also copy the platform and CRM product this repository is meant to exclude. Key missing boundaries include:

| Source dependency | Responsibility in the private host | Needed for a runnable extraction |
| --- | --- | --- |
| Agent version, Run, and event models | Immutable capability versions and execution evidence | Neutral workflow request/result and trace interfaces |
| Execution controls and budgets | Time, token, cost, and request ceilings | Injected execution-budget protocol and local fixture implementation |
| Deployment and capsule registry | Exact versions, permissions, and configuration | Small static composition loader or an external host adapter |
| Connector runtime and typed port registry | Validation, transport, correlation, retries, and receipts | In-process fixture transport using the included envelope/port contracts |
| CRM Company and Prospect types/stores | Application business records and transactions | Generic organization and lead DTOs plus repository protocols |
| Model router and settings | Provider selection, budgets, and secrets | Local-only provider configuration with no secrets in source |
| Memory persistence and review authorization | Durable scoped memory and owner review | In-memory demo store and explicit review callback |
| Search and evidence adapters | Bounded source discovery and first-party evidence | Deterministic fixture sources; no network by default |

Make it runnable by replacing these dependencies with typed ports and a neutral fixture workload, not by copying excluded product services under new names.

## Privacy and public-sharing review

The snapshot was prepared from the source working tree as it existed on 2026-10-04, including source edits that were uncommitted there. Literal organization and geography details in copied workflow source were generalized. No records, secrets, private operational traces, or local configuration were copied. Read the copied prompts and code before public use; literal replacement is not a legal, IP, or comprehensive privacy review.

This repository has no license selected yet. Review it before a public push.
