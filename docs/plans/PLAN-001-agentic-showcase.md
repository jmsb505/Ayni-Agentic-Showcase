# PLAN-001 — Separate agentic system showcase

Status: Local showcase complete; not published
Created: 2026-10-04
Last updated: 2026-10-04

## Objective

Create a standalone local Git repository beside the private Ayni source checkout. Copy only custom agent workflow and architecture material needed for an interview; leave the source repository unchanged.

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

- [x] Parse copied Python files and JSON documents.
- [x] Scan for known private organization/geography literals and common credential patterns; inspect the file inventory for CRM records and excluded application/runtime directories.
- [x] Initialize a standalone local Git history; confirm the source checkout retains only its pre-existing worktree changes.
- [x] Confirm no remote is configured; preserve the decision to keep this repository local.

## Validation and evidence

- Static Python syntax parse; no application tests or provider calls.
- JSON parse of all copied contract, fixture, and manifest files.
- File inventory and exclusion scan.
- Independent Git status and commit check.
- Source checkout status compared with the known pre-existing dirty worktree.

## Risks and decisions

Agent workflow modules depend on host services and application-owned types. The showcase is a code-review archive, not a package. Making it runnable requires a later contract extraction. Public repository license remains to be selected.

## Completion

Local validation passed: 14 Python files parsed, 15 JSON files parsed, and the targeted privacy scan found no matches. Clean local commit: 82e2461. No remote is configured. The user chose to keep the repository local. No remote was created or pushed.
