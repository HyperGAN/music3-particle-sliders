# Memory requirements

| Operation/profile | GPU count | Calculated VRAM per GPU (GiB) | Calculated host RAM (GiB) | Input limits / precision |
|---|---:|---:|---:|---|
| LM teacher/conditioning preparation | 1 | 22 | 32 | BF16; eager; one sequence ≤ 256 tokens; complete LM on one GPU |
| Native LM response training (default checkpointing=False) | 1 | 32 | 32 | batch 2, total prefix+history ≤ 256 tokens; BF16 base/FP32 trainables; eager; no checkpointing |
| Native hidden-response inference | 1 | 22 | 32 | batch 1; ≤ 256 tokens; BF16/eager; no KV; hidden response only |

These are **calculated planning budgets, not measured peaks or hard runtime ceilings**. GiB means 2³⁰ bytes. The table bounds one operation at a time; do not add sequential phases, and do add concurrently resident models/jobs. Longer contexts, larger batches, reference audio or different backends require recalculation; the package does not have a universal finite maximum for every accepted request. No benchmark, model construction, forward, GPU query or training run was performed for this document.

Frozen BF16/FP16 weights cost `2P`; FP32 costs `4P`. LoRA costs `r × Σ(in + out)`. The root particle bank is counted once (128 × 4 FP32 values). Rank-8 shared RoutedMLP branches have 6,476 parameters each: router 8→16→16→16→4 and network 12→48→48→48→8; graph variants below use this conservative branch allowance. Critic allocations are included. Frozen weights have no parameter gradients or optimizer moments.

For ParticleGAN 0.11.1 `bispf_optimized`, DualNorm has zero momentum: do **not** add Adam's two full-sized moments. We reserve **64 bytes per trainable scalar** for FP32 parameters/gradients, old/displaced parameters and normalization/constraint temporaries, including protected-normal `.double()` buffers; small scalar step/constraint state fits the workspace allowance. This is a conservative transient envelope, not 64 bytes of persistent optimizer state. No model-weight EMA copy is allocated in the supported RAW policy. A separately identified supervised AdamW comparator uses its actual two FP32 moments.

Activation estimates deliberately include FP32 routed branches and two FP32 attention-sized buffers for eager/math attention. With `L` layers, width `H`, FF width `I`, query heads `N`, tokens `T` and effective accumulated batch `B`, the uncheckpointed training allowance is `64BLTH + 12BLTI + 8BLNT²` bytes. With block checkpointing it is `8BLTH + 64BTH + 12BTI + 8BNT²`; hybrid attention uses only its full-attention layer count in the quadratic term, with recurrent state added separately. These are explicit planning allowances for saved tensors/recomputation, not a promise about every allocator/kernel. Sampling prefill uses `16BTH + 8BTI + 8BNT²`, plus actual KV/recurrent state and any materialized logits. Runtime/workspace allowances below cover Python/framework/CUDA workspaces, allocator fragmentation and the small game state; an executor must still enforce live resource checks.

Host budgets count the stated CPU loading peak, tensor caches and serialization copies plus workspace. They exclude an optional **reclaimable OS file cache** of the complete checkpoint: add the checkpoint file size if deliberately retaining that cache in RAM. Model files remaining on disk are not resident GPU weights. Training duration/update count changes checkpoint disk consumption, not this live memory calculation. RAW/FULL serialization can add CPU copies of trainables; those are included in the host allowances.

Model: **MiniMaxAI/MiniMax-Music3**, revision **`bd348f9c49ea3c1b39f33ace3436f8fad435f24e`**.

The table rounds each calculated phase sum upward to the next2 GiB. The byte ledger below gives the unrounded calculation; per-GPU split estimates are listed in device order.

| Phase | Allocation ledger (GiB before rounding) |
|---|---|
| LM teacher/conditioning preparation | frozen: 15.9898; prefill: 0.0547; workspace: 4.0000 |
| Native LM response training (default checkpointing=False) | frozen: 15.9898; trainable_transient: 0.5205; activations: 8.1562; teacher_cache: 0.0039; workspace: 6.0000 |
| Native hidden-response inference | frozen: 15.9898; adapter: 0.0325; prefill: 0.0547; workspace: 4.0000 |

Detailed arithmetic and assumptions:

- **frozen elements**: `8584475648`.
- **frozen BF16 bytes**: `17168951296`.
- **LoRA elements**: `7667712`.
- **adapted projections**: `144`.
- **generator plus critic elements**: `8731937`.
- **host loading**: `complete BF16 CPU LM + largest4,983,068,496-byte shard + two largest embedding-sized validation buffers +8 GiB + caches/export`.

Scope and limits:

- The current response API explicitly refuses acoustic/token rendering. These are real 4096-wide LM hidden-response estimates, not a promise of finished music synthesis or a Music3 DiT/VAE budget.
- The default max_sequence_length is256; Source permits up to8192. Do not reuse the256-token numbers at8192: the eager quadratic term alone is576 GiB for the accumulated batch 2/all 36-layer training envelope. Native full-checkpoint profiles are BF16, not a quantized fallback.

Source facts (read as text; no model execution):

- [music3/native_assets.py](music3/native_assets.py)
- [music3/native-assets.lock.json](music3/native-assets.lock.json)
- [music3/lifecycle.py](music3/lifecycle.py)
- [music3/response_source.py](music3/response_source.py)
- [Pinned model metadata](https://huggingface.co/MiniMaxAI/MiniMax-Music3/tree/bd348f9c49ea3c1b39f33ace3436f8fad435f24e) (architecture/parameter metadata or exact file-size lock as described above).
- ParticleGAN 0.11.1 release `d5dec2fe005305082acf382f0939a4bdb51c7131`, `recipes.py:bispf_optimized`, `optim/dualnorm.py`, `optim/direction_blend.py` and `optim/constraint_geometry.py`; shared core `routed_projection.py`, `reference.py:RoutedMLP`, `audio.py:AudioResponseCritic`. These optimizer formulas are shared dependencies, not a new optimizer implementation.

Calculated 2026-11-14. Counts, live runtime qualification and memory requirements are separate facts.
