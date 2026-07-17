# PACT V2: contract-verified receding-horizon obligations

Status: final Track 2 submission architecture. The production server imports
`PACTAgentExecutor`; the release Dockerfile copies only the PACT runtime
dependency closure. This document describes that reachable path.

## 1. Design objective

PACT separates semantic interpretation from execution authority:

- A Cerebras-hosted LLM is an **untrusted semantic compiler**. It proposes a
  small typed obligation plan but cannot execute a tool or certify success.
- A deterministic local boundary performs strict decoding, PlanIR validation,
  complete live JSON Schema validation, evidence checks, and bounded execution.
- The A2A adapter exposes at most one outstanding external operation and lets
  only evaluator-returned results enter the evidence ledger.

The implementation is domain-independent. The PACT kernel contains no
CAR-bench task IDs, public utterances, or automotive workflow branches. Domain
semantics enter through the trusted policy envelope, current user event, live
tool declarations, and returned evidence.

PACT does **not** prove that an LLM understood a natural-language policy or tool
description. Its unconditional checks begin at strict wire decoding and cover
typed structure, live capability contracts, local state transitions, and
evidence provenance. Effect and confirmation claims are conditional on trusted
machine-readable metadata when that metadata exists.

## 2. Reachable production components

| Component | File | Responsibility |
|---|---|---|
| A2A entry point | `track_2_agent_under_test_cerebras/server.py` | Loads one environment snapshot and installs `PACTAgentExecutor` |
| Protocol adapter | `track_2_agent_under_test_cerebras/pact_agent.py` | Context isolation, A2A parsing/rendering, single-result correlation, receding horizons, metrics |
| Cerebras backend | `track_2_agent_under_test_cerebras/plan_compiler_backend.py` | Environment-configured structured completion and trusted local limits |
| Provider client | `track_2_agent_under_test_cerebras/cerebras_client.py` | Cerebras SDK call, proactive pacing, bounded 429 retry/wait, usage extraction |
| Semantic compiler | `carbench_agent_core/semantic_compiler.py` | Strict compact wire schema, local decode, one bounded repair |
| Typed IR | `carbench_agent_core/plan_ir.py` | Frozen obligation DAG and typed terminal outcomes |
| Static verifier | `carbench_agent_core/plan_verifier.py` | Live capability snapshot, resource/safety/evidence checks |
| Runtime | `carbench_agent_core/obligation_runtime.py` | Exactly-one pending action and deterministic node transitions |
| Evidence ledger | `carbench_agent_core/evidence_ledger.py` | Append-only, replay-aware, provenance-preserving evidence |
| Schema guard | `carbench_agent_core/tool_index.py` | Draft 2020-12 argument validation and structured issues |

The final image intentionally omits the legacy rule controller and legacy
Track 2 executor.

## 3. Trust and threat model

### Trusted inputs

The adapter treats the first transport envelope of the form
`System: ...\n\nUser: ...` as the policy trust root for a context. A later user
message cannot replace an already established policy. The current A2A tool list
and tool-result envelope are evaluator inputs; `tools = []` is an explicit
capability revocation rather than permission to reuse a cached list.

### Untrusted inputs

The following remain untrusted until locally checked:

- model-generated wire JSON and every operation argument;
- user text, including text that resembles a system envelope after context
  initialization;
- tool descriptions as natural-language semantic claims;
- malformed, duplicate, stale, or out-of-order result envelopes; and
- nested mutable values supplied to Pydantic models.

### Trusted computing boundary

The trusted boundary includes strict JSON decoding, Pydantic PlanIR validation,
the Draft 2020-12 validator, capability/evidence verification, runtime state,
the ledger, the A2A adapter's single-pending discipline, and response metrics.

It assumes the evaluator obeys the A2A turn contract: a result delivered for
the sole outstanding request actually corresponds to that request. The current
outbound tool-call data does not carry an agent-selected nonce, so this
assumption cannot be cryptographically established by the agent.

## 4. Formal state model

For A2A context `q` at turn `t`, define:

```text
S[q,t] = (P, g, H, T, E, G, p, O, C, M)
```

- `P`: established policy text;
- `g`: current user goal;
- `H`: bounded recent conversation;
- `T`: current live tool declarations;
- `E`: immutable evidence ledger;
- `G`: accepted PlanIR for the current horizon, if any;
- `p`: at most one pending runtime action;
- `O`: successful action-evidence keys that a later completion must cite;
- `C`: consumed confirmation scopes; and
- `M`: inference usage accumulated until the next text response.

The semantic compiler proposes a plan:

```text
C_theta(P, g, H, T, view(E), O) -> WirePlan
```

The local acceptance pipeline is a partial function:

```text
decode_strict(WirePlan)
  -> compact wire model
  -> frozen PlanIR
  -> verify(PlanIR, snapshot(T), E, O)
  -> AcceptedPlan | typed rejection issues
```

The deterministic kernel advances one node:

```text
K(AcceptedPlan, E, fresh_snapshot(T))
  -> one A2A action | WAIT | terminal response | fail closed
```

## 5. Untrusted semantic compiler

### 5.1 Prompt partitioning

The compiler receives three messages:

1. a fixed domain-independent compilation contract;
2. the trusted policy serialized in its own system partition; and
3. untrusted goal, current event, bounded conversation, live capabilities,
   typed evidence, and outstanding completion obligations.

The compiler is explicitly instructed not to execute operations or claim that
an external operation succeeded. Existing evidence includes source, status,
event/call/node provenance, operation and digest fields, and exact
authorizations rather than a prose-only summary.

### 5.2 Cerebras-compatible compact wire schema

Cerebras strict structured output requires every object to set
`additionalProperties: false` and limits the response schema to 5,000
characters. Tool argument objects are dynamic and cannot safely be represented
as free-form strict-schema objects. PACT therefore uses a uniform wire node:

```json
{
  "id": "n1",
  "kind": "observe",
  "depends_on": [],
  "operation": "live_operation_name",
  "arguments_json": "{\"key\":\"value\"}",
  "success_evidence_key": "observation.1"
}
```

The provider schema:

- has a discriminated variant for each node kind;
- forbids extra fields on every object;
- constrains terminal outcomes;
- constrains operation names by enum while the schema remains at or below 5,000
  characters; and
- falls back to a string operation only if a large live inventory would exceed
  that provider limit, while the local verifier still enforces exact live
  membership.

The local decoder rejects duplicate JSON keys, non-finite numbers, a non-object
top level, wrong variant fields, non-object `arguments_json`, and any candidate
over the configured byte/character bound. The decoded argument object then
passes through the full live Draft 2020-12 schema.

### 5.3 Bounded repair

The initial candidate is decoded and verified locally. On rejection, the
compiler may receive one repair request containing:

- the rejected candidate already present in conversation context;
- value-redacted issue code and structural path; and
- a request for one complete corrected JSON object.

`max_repair_attempts` is restricted to `0` or `1`. A second invalid proposal
raises `PlanRepairExhaustedError`; it is never executed. Attempt records retain
hashes and metrics, not raw model output.

Provider termination is classified before repair. A `finish_reason=length`
response is a typed `provider_output_truncated` failure: its usage is retained,
but a repair call is skipped because the provider has not supplied a complete
candidate to correct. Empty content is likewise represented as a typed invalid
candidate and can never enter `PlanIR` or the runtime.

## 6. Typed PlanIR

`PlanIR` is a frozen, non-executable DAG. It allows only these node variants:

- `observe(operation, arguments, success_evidence_key)`;
- `ask(prompt, evidence_key)`;
- `confirm(prompt, evidence_key, authorizes)`;
- `act(operation, arguments, success_evidence_key)`; and
- `respond(text, outcome, requires_evidence)`.

All variants forbid extra fields. Arguments are JSON constants; the IR has no
callbacks, template evaluator, arbitrary expressions, imports, regular
expressions generated as code, or evidence-binding language. Dynamic values
are resolved by compiling a fresh horizon after relevant evidence arrives.

### Structural conditions

PlanIR construction rejects:

- blank or duplicate node IDs;
- unknown, repeated, or self dependencies;
- dependency cycles;
- multiple or missing response nodes;
- a nonterminal response;
- a node that does not lead to the terminal response;
- duplicate evidence producers or reuse of an input key in the same horizon;
- response evidence with no producer/input; and
- confirmation targets that are not dependent action nodes.

The verifier additionally enforces configured node and dependency-depth bounds.
The production defaults are 20 nodes and depth 12.

### Typed terminal outcomes

| Outcome | Structural/evidence condition |
|---|---|
| `completed` | Has a current action or carried evidence; cites every action success key and all outstanding prior-action keys; emitted citations are successful external evidence |
| `answered` | Contains no action and cites all observation results used in the horizon; cited evidence cannot be failed |
| `refused` | Contains no external operation |
| `declined` | Contains no action and requires an available false confirmation record |
| `fail_safe` | Contains no action; compiler contract requests failure evidence when available, but the emergency adapter path may respond without it |

The last row is intentional: the system must retain a truthful non-completion
escape path when strict decoding, provider access, or protocol parsing fails
before a typed failure record can exist.

## 7. Live capability verification

`CapabilitySnapshot.from_tools` canonicalizes each live function declaration
into:

```text
(operation, description, parameter_schema, schema_digest,
 requires_confirmation, effect)
```

It validates parameter schemas with `Draft202012Validator.check_schema`,
rejects malformed or duplicate declarations, and hashes the normalized
inventory. `ToolIndex` then validates arguments using a cached full Draft
2020-12 validator and `FormatChecker`. Validation includes nested objects,
arrays, type/enum/const, ranges, `multipleOf`, length/pattern/format,
combinators, and `additionalProperties` according to the supplied schema.

No operation is classified from its name. The trusted metadata surface is:

- `x-pact-effect = "observe" | "act" | "unknown"`; and
- `x-pact-requires-confirmation = true`.

A confirmation-required capability is forced to effect `act`. Conflicting or
invalid metadata fails closed. If effect is `unknown`, schema and membership
checks still hold, but PACT cannot guarantee whether the model's `observe` or
`act` label reflects the operation's real-world effect.

Every external emission is rechecked against a newly built snapshot. It must:

1. be a valid snapshot;
2. have the exact digest accepted at compilation;
3. still contain the operation;
4. agree with any known effect label; and
5. accept a deep JSON copy of the emitted arguments.

This closes the plan-verification-to-emission gap for capability revocation,
schema mutation, and caller-owned argument mutation.

## 8. Evidence ledger and runtime

### 8.1 Evidence records

An `EvidenceRecord` contains:

- unique evidence key and JSON-isolated value;
- source: `external` or `input`;
- status: `success`, `failure`, or `observed`;
- producer event, internal call, and node IDs;
- producer kind;
- optional operation and evaluator call ID;
- optional context, plan, argument digest, and schema digest; and
- zero or more exact `ActionAuthorization` scopes.

The ledger is append-only by evidence key. Re-adding an identical record is
idempotent; a different record for the same key is rejected. Because frozen
Pydantic objects are not recursively immutable, the ledger JSON-round-trips
records when storing and reading them. Mutating either an inbound event payload
or a returned value cannot rewrite stored evidence.

### 8.2 Event idempotency

Every runtime event is registered with a canonical SHA-256 fingerprint. A
repeated event ID with identical content is `duplicate`; reuse with different
content is `conflict`. Both leave prior evidence unchanged. An event for no
pending call or the wrong internal call is `out_of_order`.

### 8.3 One pending obligation

Node state is one of `pending`, `waiting`, `satisfied`, or `failed`. The runtime
walks the stable topological order and selects the first pending node whose
dependencies are satisfied. On Ask, Confirm, Observe, or Act it allocates a
unique internal call ID, marks that node waiting, and emits exactly one action.
While any node is waiting, `step()` returns no second action.

An external result can satisfy a pending Observe/Act only after matching:

```text
internal call ID
context ID
plan ID
node ID
operation
argument digest
capability/schema digest
non-empty evaluator call ID
explicit SUCCESS status
```

Explicit `FAILURE` creates `<success_key>.failure` evidence, marks the node
failed, and blocks the plan. Malformed or status-less result envelopes are
rejected by the adapter and cause safe termination rather than success.

## 9. Receding horizons and outstanding actions

PACT avoids a model-generated expression language by replanning after evidence
becomes available:

- Ask response -> record input -> compile a fresh horizon;
- positive confirmation -> record scoped authorization -> fresh horizon;
- observation result -> record external success -> fresh horizon;
- successful action -> record success and advance a still-valid accepted graph;
  if a later observation or user interaction occurs, subsequent compilation
  sees that action evidence.

Every successful Act adds its success key to
`required_completion_evidence`. The compiler receives that set, and the
verifier requires a later `completed` terminal to cite every member. A carried
key is valid only if the ledger identifies it as a successful external action
record. If later work fails, the adapter reports partial failure rather than
forgetting that an effect already occurred.

## 10. Scoped one-shot confirmation

Within one plan, a Confirm node lists exact descendant Act node IDs. When the
user explicitly confirms, the ledger stores an authorization for each target:

```text
(schema_digest, operation, sha256(canonical_arguments))
```

A future horizon may use a prior confirmation only when all three fields match
and the plan declares that confirmation record as an evidence input. The
adapter removes the matching scope from the compiler-visible projection after
the authorized Act succeeds. A confirmation therefore cannot authorize a
different operation, different arguments, changed schema, or a second use.

Confirmation parsing is deliberately narrow. Unambiguous positive/negative
phrases become booleans; mixed or ambiguous text produces a yes/no retry rather
than an action. A negative decision is recorded as failure-status input and
terminates with a non-completion acknowledgement.

This guarantee applies only when a live capability is explicitly marked
confirmation-required. PACT does not infer consequentiality from an operation
name or natural-language description.

## 11. A2A correlation boundary

The A2A adapter accepts exactly one result for exactly one outstanding external
obligation. Before constructing the trusted runtime event, it requires:

- one result object, not a batch;
- the pending operation name; and
- a nonempty evaluator-assigned `tool_call_id`.

It then binds that result to the pending runtime's plan/node/argument/schema
metadata and uses the evaluator ID plus content in the event fingerprint.

The current outbound `ToolCallsData` lets the agent send operation and
arguments but not an agent-chosen call nonce that the evaluator must echo. As a
result, PACT cannot independently prove that a returned evaluator ID belongs to
the exact outbound call or distinguish a replay of an identical call by a
misbehaving evaluator. The single-pending rule removes local ambiguity but does
not remove this protocol trust assumption. Reports and proofs must preserve
this distinction.

## 12. Inference and rate-limit bounds

Each semantic compilation performs:

```text
1 initial completion + at most 1 verification-guided repair
```

Therefore successful semantic completions per compile request are at most two.
The following independent limits are enforced locally and configurable:

- policy, goal, current-event, conversation, and total context characters;
- conversation message and evidence-record counts;
- candidate size;
- PlanIR node count and dependency depth; and
- maximum completion tokens.

Provider rate-limit retry is a separate bounded loop. The Cerebras client has:

- finite `PACT_CEREBRAS_MAX_RATE_LIMIT_RETRIES`;
- finite `PACT_CEREBRAS_MAX_RATE_LIMIT_WAIT_SECONDS`;
- proactive token pacing using recent quota headers;
- reactive `Retry-After`/reset-aware scheduling with bounded queue backoff; and
- `CerebrasRateLimitBudgetExceededError` when the next retry or wait would
  exceed a configured budget.

No provider path waits or retries indefinitely.

The submitted Track 2 defaults are `gpt-oss-120b`, low reasoning effort, and a
4,096-token completion ceiling. These are environment-configurable, but the
low setting is deliberate. A high-reasoning canary consumed 12,861 prompt,
4,096 completion, and 4,093 thinking tokens before returning
`finish_reason=length` with no content. The production classifier now records
that failure and its usage without spending the single repair allowance on a
known-truncated response.

## 13. A2A metrics

Every A2A text response includes `metadata.turn_metrics`, including asks,
confirmation prompts, typed terminal responses, refusal, and safe-error text.
Tool-call responses defer metrics so usage accumulates across the external
operation loop until the next text response.

Reported fields include:

- `prompt_tokens`;
- `completion_tokens`;
- `thinking_tokens`;
- `num_llm_calls`;
- `avg_llm_call_time_ms`;
- `num_passes`;
- `quota_wait_time_ms`;
- model and provider-reported cost.

Compiler rejection and repair attempts retain their usage. If a typed compiler
error exposes completed attempts, the safe terminal response reports those
metrics before the context accumulator resets.

## 14. Enforced invariants

These statements assume the production adapter is the only emission path, the
hash function and JSON validator behave correctly, and the live schemas are
valid.

### I1. Structural plan validity

Every executed plan is a bounded DAG with unique IDs/evidence producers,
exactly one terminal response, and no node outside the response ancestry.

### I2. Capability confinement

Every emitted external operation belongs to the current digest-matched live
snapshot, and its fully materialized arguments validate against that
operation's current Draft 2020-12 schema.

### I3. Single outstanding effect

The runtime emits no second Ask, Confirm, Observe, or Act while one action is
waiting for its compatible event.

### I4. Evidence non-escalation

A failed, malformed, duplicate, conflicting, out-of-order, wrong-kind, or
wrong-provenance event cannot create successful evidence or satisfy the pending
node.

### I5. Grounded completion

A typed `completed` response cites successful external evidence for every
current Act and every carried successful-action obligation. Input-only,
failure, or missing evidence cannot discharge completion.

### I6. Conditional scoped confirmation

For a capability explicitly marked confirmation-required, an Act has either a
current authorizing Confirm ancestor or an unconsumed true confirmation record
matching schema digest, operation, and argument digest.

### I7. Context isolation

Policy, goal, tools, ledger, runtime, obligations, authorization consumption,
and metrics are keyed by A2A `context_id`; cancellation removes state and its
lock together.

### I8. Bounded semantic inference

One compile request has at most two successful completion calls. Provider
rate-limit retries and cumulative wait are separately finite.

## 15. Failure policy

PACT fails closed at each boundary:

| Failure | Visible behavior |
|---|---|
| Invalid/missing policy event or empty goal | Generic safe text |
| Invalid/removed capability or schema | No call; safe text |
| Invalid wire/PlanIR | One bounded repair, then safe text |
| Provider/configuration/rate-limit budget error | Safe text with available attempt metrics |
| Malformed/multiple/unmatched tool result | No success evidence; safe text |
| Explicit tool failure | Failure evidence; no completion claim |
| Failure after an earlier successful Act | Partial-failure text |
| Ambiguous confirmation | Ask for an explicit yes/no answer |
| User message while an external result is pending | Preserve obligation and report waiting |

Generic failure text intentionally reveals neither verifier internals nor
provider exception data.

## 16. Reproducible engineering evidence

At the report freeze on 17 July 2026:

- `uv run pytest -q` passed 330 tests plus three subtests;
- 168 collected tests directly covered PlanIR, verifier, semantic compiler,
  evidence ledger, obligation runtime, PACT A2A adapter, ToolIndex, and bounded
  Cerebras retry behavior; and
- the only warnings came from vendored evaluator Pydantic deprecations.

Focused coverage includes graph cycles/limits/reachability, strict wire-schema
size and shape, one-repair exhaustion, complete nested JSON Schema constraints,
schema revocation, deep mutation, external failure, replay/conflict/order,
wrong operation/digests, confirmation scope and consumption, carried completion
obligations, context isolation/cancellation, safe-error metrics, and finite
rate-limit budgets. Generic non-automotive tool schemas exercise the same
kernel.

These are deterministic engineering tests. They are not a CAR-bench reward,
hidden-set estimate, or three-trial consistency result. Earlier workflow-agent
scores are not evidence for PACT V2 and are intentionally excluded from the V2
technical report.

One public-train `base_0` smoke run exercised the complete A2A loop with the
submitted low-reasoning default. It obtained reward 1.0, with
`r_actions`, `r_tool_subset`, `r_tool_execution`, `r_policy`, and
`r_user_end_conversation` each equal to 1.0. The three-user-turn trace performed
a state lookup, confirmed and executed the shade action, obtained weather,
confirmed the rain-dependent action, set the sunroof to 50%, and completed from
recorded evidence. Seven compiler completion calls (including local repair)
reported 97,922 prompt, 3,972 completion, and 2,032 thinking tokens, for 103,926
total tokens; quota pacing accounted for 360,753.4 ms. Every observed provider
response ended with `finish_reason=stop`, and the run had no missing-tool,
execution, or policy error. This is a single end-to-end closure check, not a
performance estimate or evidence about the hidden set.

## 17. Release checklist

Before final submission:

1. Rebase/verify compatibility with the latest official main branch.
2. Run the complete unit suite, Ruff, and mypy.
3. Repeat the completed public-format A2A smoke against the exact candidate
   image.
4. Build the final image for `linux/amd64` from the selective Dockerfile.
5. Scan source, image layers, and scenario TOML for secrets.
6. Publish the GHCR package publicly and pin its immutable digest.
7. Validate the exact digest with README option C.
8. Propagate the same digest to the submission scenario and form.
9. Re-run the four-page report build and update only dated test evidence that
   changed; do not add unverified task scores.

## 18. Known limitations

- Natural-language policy and goal semantics remain model-produced and are not
  formally verified.
- Unknown effect metadata cannot establish Observe-versus-Act correctness or a
  confirmation requirement.
- Exact evaluator result origin relies on the A2A single-pending contract
  because outbound agent nonces are unavailable.
- The ledger is process-local and is deleted on cancellation; it is not a
  durable distributed audit log.
- SHA-256 provenance is content addressing, not a signature or remote
  attestation.
- One repair bounds compute but may trade away recovery on semantically hard
  requests.
- The engineering suite establishes mechanisms, not benchmark task quality;
  final competitiveness must be measured separately without changing the
  kernel toward benchmark-specific branches.
