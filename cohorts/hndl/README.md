# Music3 HyperGAN/HNDL candidate

This separately packaged CPU integration uses core 0.12 at
`a794e3b4a8813b012f2c4faa3da82b78d4a90214`, official PG 0.9.4, actual clean
HyperGAN develop `5f368a3770cf5bdae0448995c9f76e29af8049f7` and HNDL 0.10.0.
It binds the existing shared HNDL routed projection to the genuine Qwen3
attention q/k/v/o linears, using one owned 128×4 bank, rank 8 and alpha 8.
The AR readout retains the existing linear/SiLU/linear architecture, executed
as an HNDL graph through HyperGAN's verified builder. The shared
`ConditionalPG09` owns all losses, optimizers, updates, prior and checkpointing.

Install only in a separate Python 3.11–3.14 environment. The final coordinated
source pin is `a794e3b4a8813b012f2c4faa3da82b78d4a90214`; version 0.12.0 alone
does not admit a different commit. The recursively generated source/data map in
`music3_hndl/core.lock.json` binds 51 files, the exact VCS identity, and the owner's
canonical reference wheel SHA256. The tested public VCS build has its own wheel
archive digest and must match all 51 source/data bytes. Build this candidate with
`python -m build --no-isolation cohorts/hndl`, then install its wheel in that
already qualified environment. This candidate does not install the historical
Music3 package or change its core 0.8.2/PG 0.9.1 pin.

The previous 4abc preview remains an old cohort in its original receipts and
packets. It is not relabeled: actual old preview training and raw-export packets
are refused without live mutation by the final candidate. See
[FINAL-CORE-PIN.md](FINAL-CORE-PIN.md) for the exact pin, install commands, artifact
identities and current tests. Earlier review notes remain historical evidence.

Both product manifests contain direct VCS dependencies. PyPI does not accept
those dependencies; these artifacts must not be uploaded to PyPI. Their use is
source installation and local wheel qualification only. No package publication
is authorized or performed.

Source checks have two owners. `music3_hndl.source.verify_core` binds the exact
core wheel/VCS provenance, complete installed core source/data inventory and
import origins. Shared core's `fibo_hndl.verified_builder` then requires
HNDL 0.10.0, verifies every HNDL Python source, the six-file HyperGAN builder
closure, and clean build provenance for commit `5f368a3770cf5bdae0448995c9f76e29af8049f7`.
The factory calls it before Qwen3 allocation, and projection/critic construction
also use it. Product graph bytes are recorded in the critic identity and signed
by the shared checkpoint architecture. No check is replaced by a version pin.

`scripts/reproduce_inventory_cpu.sh` is an **owner-machine-only receipt replay**.
Its absolute defaults refer to the original executor; dependency binaries are
reused through read-only, unpinned symlinks. It is not portable clean-install
evidence. Input locations, including `HNDL_CORE_WHEEL` and `HYPERGAN_CHECKOUT`,
can be overridden. Separate public-source installation evidence is recorded in
the final review receipt.

Closing always attempts both cleanup owners and restores native modes/gradient
flags. If a foreign forward remains attached, `cleanup_pending` stays true and
the failed ledger remains reachable. The game and slider are closed and unusable:
game access and adapter response calls raise before executing the host. Keep raw
model/adapter references only to resolve ownership and retry cleanup; restoring
modes does not certify that a conflicted host is clean or safe to reuse.

```python
from music3_hndl import build_cpu_game

with build_cpu_game() as run:
    run.game.step()  # actual shared ParticleGAN update
    checkpoint = run.game.state_dict()
```

`build_cpu_game` exposes a bounded random CPU width 32 Qwen3 fixture, one to four steps,
and explicit seed/checkpointing/preset. Native width 4096 is rejected by the
shared continuous-domain admission gate before model allocation. No pretrained
loader, model download, AR sampler, decoder or acoustic output is exposed.
The fixture's vocabulary is synthetic; it does not qualify native tokenization.
This product API exposes raw game checkpoints/exports and no session EMA option;
`ema_decay` is explicitly refused. Upstream synthetic CUDA fixture results do not
admit Music3 native width, pretrained loading, audio rendering or quality.

This is a fresh checkpoint cohort. Keep historical environment, source locks
and checkpoint identities. Old routed exports are not ordinary LoRAs; this
candidate does not claim ordinary-LoRA distillation or ComfyUI qualification.
Matched-weight parity, particle gradients, exact-zero behavior and complete
shared checkpoint restore must pass on installed packages before adoption.
No package/model release, deployment or GPU work is triggered by this source.
