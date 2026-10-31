# Ordinary LoRA compatibility

The routed particle export is nonlinear and cannot be renamed into an ordinary
LoRA. `music3.distill.distill_lora` collects explicit token-context activations
from an already open Music3 response adapter and calls the installed shared
`particle_sliders.distillation.fit_routed_down` implementation. It fits the
routed bottleneck and retains each teacher up matrix and alpha. It introduces
no training objective, optimizer or GAN update in this product.

```python
from music3.distill import distill_lora
from particle_sliders.lifecycle import SessionConfig
from particle_sliders.response import open_response

config = SessionConfig("music3", "my-new-cpu-fixture", dummy=True,
                       width=32, steps=2, batch=2, seed=7)
with open_response(config) as run:
    run.train()
    factors, provenance = distill_lora(
        run.game.backend, [[11, 13, 17], [29, 31, 37]], max_rows=128)
```

This example trains only a tiny random CPU software fixture. The helper accepts
1..128 contexts, each at most 256 tokens and within the model's native budget,
and at most 512 calibration rows per target. Collection is deterministic in
caller order and never truncates a token context. Forward hooks are removed on
success or failure. Teacher state, modes, gradients, forwards and CPU RNG are
preserved. Returned factors are detached CPU tensors; no file is published.

The factors use ordinary native `LoRANetwork` names such as
`lora_te-model-layers-0-self_attn-q_proj.lora_down.weight`, with separate
q/k/v/o projections and no particles or bridge tensors. They load strictly into
the shared `LoRANetwork` configured with rank8, alpha8, delimiter `-`, prefix
`lora_te`, `target_replace=["Qwen3Attention"]`, and `train_method="full"` on a
fresh matching frozen host. They are not PEFT or converted ComfyUI keys. Keep
the provenance alongside any locally saved factors: it records the teacher
adapter digest, frozen-base identity, exact core commit, calibration digest,
factor digest, fitter, row bounds and per-target calibration residuals.

This is a separate approximation, not a resumable training checkpoint. Strength
0 bypasses the ordinary LoRA exactly; +1 is the teacher treatment being
approximated. Calibration residuals do not establish held-out quality. Native
width4096, pretrained distillation, token/acoustic sampling and ComfyUI output
remain unqualified and refused by this helper. Historical published LoRAs keep
their existing identities and converter instructions in [COMFYUI.md](../COMFYUI.md).
