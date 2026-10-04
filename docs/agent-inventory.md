# Agent workflow inventory

The core path is a staged business workflow. These are bounded capabilities, not autonomous chat sessions that delegate freely to one another. The application chooses a stage and owns its business transaction; Ayni Core owns the shared execution controls.

| Capability | Source in this repository | Purpose | Decision and review boundary |
| --- | --- | --- | --- |
| market.research | source/app/company_brain.py and source/app/market_research.py | Build an operating-context profile and market strategy from supplied context and bounded research. Keep context, source observations, strategy, and findings distinct. | Deterministic schema and citation checks; Judge may assess readiness; an operator reviews and approves the profile. |
| lead.discovery | source/app/discovery_autopilot.py | Plan bounded search, screen and qualify candidates against an approved target profile, detect duplicates, and queue candidates for review. | Deterministic source and eligibility rules; Judge may assess fit; a person normally retains or archives candidates. |
| lead.research | source/app/prospect_research.py | Build a cited company and fit assessment for a retained candidate, including unknowns and risks. | Deterministic source validation; Judge may assess research quality; the assessment remains reviewable by a person. |
| lead.proposal | source/app/proposal_drafting.py | Draft a proposal and related outreach text from approved qualification and research. | Deterministic claim and delivery constraints; Judge may assess draft quality; the draft is not sent automatically. |
| lead.outreach | source/app/delivery.py | Keep delivery as a separate capability and preserve the exact approved recipient and content. | Human approval is bound to the effect; provider evidence must verify the result, and uncertain delivery requires reconciliation. |

The source capability catalog also contains a read-only platform assistant and a legacy CRM company-description capability. Their product-facing adapters are outside the requested boundary. The CRM agent application under the imported CRM tree is upstream, depends on CRM services, and is not part of this custom workflow snapshot.

## Stage order

    approved organization context
        -> market.research
        -> human profile review and approval
        -> lead.discovery
        -> human candidate review
        -> lead.research
        -> lead.proposal
        -> human proposal review
        -> separately approved lead.outreach action, if configured

The workflow may stop at any stage when evidence is missing, a provider is unavailable, a budget is exhausted, validation fails, or a person needs to review. Missing Judge output does not become an implicit positive decision. Search results or a generated hypothesis do not establish customer demand.

## What is and is not an agent

- A workflow stage is an agent capability because it has a versioned input/output contract and bounded reasoning work.
- The Judge is a shared decision provider. It is not an extra workflow stage or an independent authority.
- Search, memory, model, and action adapters expose capabilities. They do not decide the next business stage.
- A Run queue and multiple workflow stages do not imply a parallel swarm. The current local model path serializes generation; stage orchestration remains application-controlled.
