# Execution ports

Execution ports use the shared connector envelope and retain their own typed request and result contracts. The envelope carries exact capsule and port versions, tenant and actor context, correlation, operation identity, and effect class. It does not decide whether a port call is authorized.

| Port | Capability | Effect |
| --- | --- | --- |
| harness.execute | Prepare a bounded generation request from an agent version, input, schema, and any validated tool result | none |
| model.generate | Generate content from prompts and return provider and usage metadata | none |
| tool.invoke | Call a tool allowed by the immutable agent version and its declared limits | read |
| action.apply | Apply a separately authorized and approved external effect | consequential |
| judge.evaluate | Assess bounded state against typed Choice, Score, or Noul questions | none |

Simple JSON and LangGraph JSON implement the same harness port. Bounded JSON uses the same generation and tool ports while the host applies its validation loop. Provider adapters expose compatible generation behavior; a deployment binds the selected provider version.

The connector validates capsule and port pins, payload shape, effect class, and execution context. The receiver still enforces authority. Tool calls remain within the versioned allowlist. Action success requires a provider receipt; an uncertain result requires reconciliation before retry.

The host registry, deployment resolver, authorization checks, connector dispatch, Run evidence, and effect recovery are omitted from this showcase.
