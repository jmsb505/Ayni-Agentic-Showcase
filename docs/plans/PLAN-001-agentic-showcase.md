# PLAN-001 — Separate agentic system showcase

Status: In progress
Created: 2026-10-04
Last updated: 2026-10-04

## Objective

Create a standalone public-facing Git repository beside the private Ayni source checkout. Copy only custom agent workflow and architecture material needed for an interview; leave the source repository unchanged.

## Scope and exclusions

Include the custom Ayni workflow source, harness/provider interfaces, memory schemas and retrieval, Judge logic, generic connector contracts, selected capsule manifests, and explanatory documentation.

Exclude the CRM product, Ayni Core runtime, persistent infrastructure adapters, private deployment files, credentials, records, logs, screenshots, and upstream CRM agent application. Do not rewrite source modules in the private checkout.

## Milestones

### M1 — Curate and explain the boundary

- [x] Inventory source ownership and module dependencies.
- [x] Copy and generalize selected source files.
- [x] Document the agent stages, harnesses, capsules, connectors, Judge, and memory.
- [x] Explain why this source snapshot is not yet runnable.

### M2 — Verify the independent repository

- [ ] Parse copied Python files and JSON documents.
- [ ] Scan the repository for private organization/geography literals, credentials, CRM records, and excluded application/runtime directories.
- [ ] Initialize a clean local Git history, then check source checkout status for unintended changes.
- [ ] Confirm no remote is configured until the curated snapshot is ready for public sharing.

## Validation and evidence

- Static Python syntax parse; no application tests or provider calls.
- JSON parse of all copied contract, fixture, and manifest files.
- File inventory and exclusion scan.
- Independent Git status and commit check.
- Source checkout status compared with the known pre-existing dirty worktree.

## Risks and decisions

Agent workflow modules depend on host services and application-owned types. The showcase is a code-review archive, not a package. Making it runnable requires a later contract extraction. Public repository license remains to be selected.

## Completion

Pending validation and the initial local commit.
