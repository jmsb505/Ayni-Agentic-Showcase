# Judge evaluation port

The Judge is a provider-neutral semantic assessment capability. A workflow sends bounded state and named questions through the typed evaluation port. The Judge returns answers and provider usage metadata; workflow code owns the decision policy.

| Type | Meaning | Result |
| --- | --- | --- |
| Choice | Select one option from the complete set supplied by the caller | Selected option, optional distribution, optional confidence |
| Score | Place the subject on a caller-defined ordered scale | Numeric score, optional level distribution, optional confidence |
| Noul | Estimate whether one condition is true | Probability from 0 to 1 |

Requests use stable question identifiers, clear instructions, and evidence within the caller's authorized scope. Choice criteria are mutually exclusive. Score criteria describe concrete ordered levels. Use separate Noul questions for independent conditions.

The response must answer the submitted question identifiers using matching answer types. Choice answers must match a declared option; scores must stay within the declared range. Confidence and probabilities are model signals, not permissions or proof of correctness.

The Judge does not set thresholds, change workflow state, approve records, grant access, write memory, authorize effects, or reconcile uncertain outcomes. Workflow code applies a versioned rule to its typed answer. Deterministic checks remain in code; high-impact or unresolved decisions go to a person.

Bound calls by question count, state size, latency, and cost. Do not retry automatically. If the provider is disabled, unavailable, malformed, uncertain, or over budget, use a declared fallback or ask for human review.

Approved memory may be included as clearly labeled context. A Judge answer does not become memory; a proposed memory remains a candidate until an authorized reviewer approves it.
