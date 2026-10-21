# Scheduled Music3 response conformance slot

Status: planned, not executed. Parent owns resource admission and the exact
installed environment. Use a fresh output directory and the reviewed exact
core0.8.2/official PG0.9.1 wheels, canonical38-file lock and companion archive.
The current exact core source is `f6807a91546e43e198645a3f8925d2dc7c9e03b4`,
with corrected canonical lock SHA256
`12e2310b2680173b5a6c3c220270d3d79cb9029d5660bf23f7d6e6f06fd986f3`.
The legacy CLI/core0.6.0 environment remains untouched. Do not install native
dependencies or build a product wheel before the slot is authorized.

CPU only, offline, one operator/interop thread, CUDA hidden/uninitialized, no
pretrained files. Use the parent's resource caps, memory guard and timeout.
Build/install the product only in the admitted isolated target, then execute
from `/tmp` with `python -I` and installed product metadata/entry points.

The first authored selection is `tests/test_response_proposal.py`:22 cases
from10 test definitions. It covers product cleanup nesting, duplicate wrapper
refusal, sibling release, forward/marker/LoRA conflicts and explicit retries;
both response presets' exact split resume; raw/EMA v1 and separate data-free v2
inference; setup-failure cleanup; native-width/profile/media refusal; registry/
lock identity; and invalid restore. No collection or execution occurred.
Run only this new module, not the historical old-core product suite.

Initial scheduled bound: serial CPU, both Torch thread counts1, offline and
CUDA hidden, hard2GiB process/group/address cap and16GiB host reserve, with a
90-second worker deadline. The parent must confirm these limits and host/pilot
availability before execution. Stop and report any guard failure; do not retry
with relaxed limits or overlap the MOSS pilot. Save installed origins, resource
telemetry, XML and exact artifact/source hashes. Broader CLI/fresh-process/
injected-observation cases below remain a subsequent explicitly bounded phase.

1. Verify the installed product commit/source hashes, response/inference entry
   points and every canonical core file. Reject altered/missing/extra core
   sources, foreign imports, wrong dependency lines and editable installs.
2. Open the registered response backend with a32-wide random Qwen3 fixture,
   each BISPF preset and checkpointing on/off. Verify native shape, neutral
   student conditioning, shared history, frozen base and gradients/updates to
   the actual128x4 bank. Check optimizer ownership by parameter identity.
3. Compare four uninterrupted updates with two plus two in fresh processes,
   exactly: every state tensor, metric, EMA value, optimizer history, prior/
   penalty state, coverage, private RNG and next seeded numerical response.
4. Corrupt cohort/source/profile/shape, optimizer history, EMA and packet kinds.
   Reject before mutation and show the original live session remains usable.
   Old CLI, image-session and old Music3 packets must not enter the new route.
5. Verify provider identity and frozen target/condition bytes survive mutation
   of returned rows/conditions and repeated deterministic evaluation.
6. Verify exact base versus strength0, supported0.5/1, and refusals for negative,
   >1, nonfinite strengths, rendering, references and media output requests.
7. Snapshot full training packet, ambient RNG, adapter buffers/modes/gradients,
   native nonpersistent buffers/RoPE counters and native flags/gradients. Exercise
   repeated observations plus provider bounds/fetch/forward failures. All
   observation state must restore, and invalid bounds must fetch no row.
8. Export/load v1 raw and EMA adapters in matching new sessions. Verify public
   adapter export is separate from training resume and has the expected scale.
9. Export v2 raw/EMA, remove training data/configuration, forbid training factory,
   provider, critic/game/optimizer construction, and open in a fresh process.
   Probe genuinely new IDs and compare to the corresponding trained selected
   network. Reject resume from v2 and reject runtime training-data overrides.
10. Inject native construction/attachment/teacher/provider/critic failures and
    update failures. Check exact forward/checkpoint restoration and idempotent
    cleanup. A failed observation must leave training usable.
11. Exercise shared CLI capabilities/train/resume/evaluate, both artifact
    formats and native-only response infer. Check explicit file names, raw/EMA
    origin/hash receipts and no waveform/render claim or unintended output on
    preflight refusal.
12. Before any Qwen3 constructor, tokenizer/model reader or device allocation,
    refuse width4096 native requests, unavailable overlay/profile combinations,
    source/shape mismatch, transformer/encoder hosts, staged objects and wrong
    precision/device. Confirm CUDA stays uninitialized throughout.

Record exact installed source map and product wheel/source hashes, commands,
thread/resource settings, XML, full/split equality receipts and failure gates.
These checks qualify software wiring only. They do not admit pretrained/GPU,
native tokenizer/history, acoustic sampling/listening or HNDL matched-weight work.
