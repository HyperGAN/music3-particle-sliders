# Checkpoint-ownership adoption source pass

Parent-routed exact shared source:
`f6807a91546e43e198645a3f8925d2dc7c9e03b4` on PG0.9.1/core0.8.2.
The canonical lock was copied unchanged and its corrected SHA256 is
`12e2310b2680173b5a6c3c220270d3d79cb9029d5660bf23f7d6e6f06fd986f3`.
It binds the ownership qualification receipt and all four guides, matching
`task-2/evidence/checkpoint-ownership-packaging/metadata-correction.json`.
Exact runtime source, wheel, sdist and source/docs archive were verified as
inert bytes. No artifact was installed or loaded as executable runtime code.

This updates only the new response proposal. Historical core0.6.0 lock,
requirements, CLI/scripts and cohorts remain byte-identical to PR5. Prior
source receipts at f64874b and0373c98 remain evidence for their original heads;
neither is relabeled as qualification for this new source. Version0.8.2 is
shared by different unpublished candidates, so commit/map/lock/wheel identity
is mandatory. No dependency-line downgrade or checkpoint migration occurred.

## Ownership sequence

Music3ARResponseAdapter owns the shared AdapterScaleCheckpoint handles created
for selected Qwen3 blocks. Product cleanup first calls close_checkpointing(),
which attempts all handles in reverse order, releases successful siblings,
and retains failed handles for retry. Only after that succeeds does the product
restore its own attention-linear LoRA forward attributes, then call shared
native_cleanup for modes, gradient flags and checkpoint configuration.
The former blanket block-forward/marker restoration is removed.

Product LoRA cleanup checks that its installed forward functions still own
the targets. It refuses a foreign replacement instead of overwriting it.
Successful forward restoration is tracked independently, so a later broader
cleanup error can retry without restoring those forwards again.

A cleanup refusal raises CleanupPending with the live adapter and a
retry_cleanup callback, retaining the shared failed-handle ledger even during
factory setup/update error propagation. The product does not use a disposable
ExitStack as a retry ledger and does not restore broader native state in finally.
Resolve the external forward/marker owner first, then call retry_cleanup().
The Binding.close callback is also directly retryable/idempotent. Common
session close methods mark sessions closed before invoking their callback;
retry the retained callback, not the already-closed session's close method.
No common close-policy rewrite is included.

The native-only response factory still constructs only base/adapter, with no
provider, teachers, critic, game or optimizer. It uses the same cleanup owner
and native observation hook. Its normal adapter installs no training checkpoint
wrappers. v1 session artifacts remain separate from v2 inference artifacts.

## Authored conformance and remaining hold

`tests/test_response_proposal.py` adds22 pending cases across10 definitions.
The ownership cases exercise nested shared handles, same-block conflict,
successful sibling release, external forward/marker replacements, product LoRA
replacement, retained-adapter retries, native restore ordering and setup failure.
Other cases exercise exact response split resume under both public presets,
raw/EMA v1 versus v2 inference separation, fresh response IDs, state-preserving
observation, registry/lock identity, invalid restore and before-factory refusals.
They are source-authored only; not collected or executed.

The bounded installed-product plan is in response-session-test-plan.md.
Execution remains pending parent resource admission while the actual MOSS pilot
is active. Native Music3 width4096, unavailable continuous-domain overlays,
pretrained/tokenizer/history and acoustic/waveform operations remain refused.
No recovered native-profile support or completed product qualification is claimed.
Matched-weight HNDL migration remains separate. No new shared interface gap was
identified by this source pass; retained callback retries follow the documented
shared session-close policy.
