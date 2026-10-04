# Memory and retained context

The system uses three memory tiers with different roles and trust rules. Retrieval is scoped to the active deployment and, where requested, the agent or entity. Unreviewed candidates are not injected as approved context.

| Tier | What it stores | How it becomes usable | What it is not |
| --- | --- | --- | --- |
| Semantic | Subject/predicate/value assertions, source references, confidence, validity, revision, and optional entity relationships | A candidate must be approved by the owning operator. Current approved records are eligible for retrieval. Superseded or out-of-date facts are filtered. | A Judge answer, search snippet, or unreviewed model claim is not an approved fact. |
| Episodic | A short summary linked to a completed or failed Run, agent, deployment, time, and source references | It can be included when a retrieval request explicitly allows episodes. It remains tied to its originating Run. | It is not the audit trail and does not prove the summary is true. |
| Procedural | Evidence-backed instructions or playbooks, optionally scoped to one agent | A human approves or rejects each candidate. Only approved instructions in the matching deployment and agent scope are returned. | It cannot override higher-priority instructions, permissions, or workflow policy. |

## Related data that is not one of the three tiers

Knowledge collections contain source documents used for retrieval and citations. They are bounded by deployment and an allowed collection list. A citation keeps source identity and an excerpt so a claim can be checked. Retrieved source content is evidence for a particular claim, not durable agent memory by itself.

Run history stores operational evidence: input/output, ordered execution events, model and Judge activity, versions, errors, and effect receipts. It is the record of what happened. A Run summary may later become an episodic memory, but the episode does not replace the Run.

Organization context supplied to a workflow is application-owned input. Human approval may promote a reviewed profile into semantic memory, but a generated draft does not overwrite the baseline. These application records are described conceptually and are not included in this repository.

## Retrieval and trust

- Filter by deployment and any agent/entity/predicate scope before ranking.
- Enforce result-count and character limits so a memory lookup cannot grow a prompt without bound.
- Return only approved, current semantic assertions and approved procedures.
- Keep episodes opt-in and visibly distinct from factual assertions.
- Surface contradictory facts instead of silently choosing one.
- Label memory as context. Do not treat it as public evidence, an instruction override, or proof of authority.
- Record memory identifiers and versions in the host trace so an operator can inspect the source under the normal access rules.

The copied source files show memory schemas and an in-memory store excerpt. Persistent storage, human authorization, and the Run evidence service are host responsibilities and are not included here.
