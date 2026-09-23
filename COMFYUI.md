# ComfyUI: MiniMax Music 3 sliders

Published adapters are on
[ntc-ai/minimax-music3-concept-sliders](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders).
The September 16, 2026 release is `uni16-fresh-selected-v2`: 16 native
exports and 16 ComfyUI exports converted from those checkpoints. This
repository does not vendor the base model or the adapter weights.

There is no custom node in this repo. Load the converted file with stock
**Load LoRA**. A product-specific Comfy node is not part of this skeleton.

## Language-model sliders (the published 16)

1. Download `comfyui/uni16-fresh-selected-v2/<id>_step<N>_comfyui.safetensors`
   into `ComfyUI/models/loras/`.
2. Use **Load LoRA** with the Music 3 text encoder connected to **CLIP**.
3. Set **strength_model = 0** and **strength_clip = 1** for the published On
   treatment. Compare against CLIP strength 0. Start with one adapter.

The text encoder needs separate `q_proj`, `k_proj`, `v_proj`, and `o_proj`
layers. A merged `qkv_proj` checkpoint has no module for the separate q/k/v
factors, so those keys log `lora key not loaded`. `o_proj` still applies.
Converted keys look like:

```text
text_encoders.model.layers.N.self_attn.q_proj.lora_A.weight
```

## Transformer sliders

Shipped transformer LoRAs use LoRANetwork names
(`lora_unet-transformer_blocks-N-attn-to_q.lora_down.weight`), not PEFT.
After conversion, q/k/v are fused into ComfyUI's `to_qkv` linear:

```text
diffusion_model.diffusion_transformer.transformer.layers.N.self_attn.to_qkv.lora_A.weight
diffusion_model.diffusion_transformer.transformer.layers.N.self_attn.to_qkv.lora_B.weight
diffusion_model.diffusion_transformer.transformer.layers.N.self_attn.to_qkv.alpha
diffusion_model.diffusion_transformer.transformer.layers.N.self_attn.to_out.lora_A.weight
```

Put the file in `ComfyUI/models/loras/` and load it on the MiniMax Music 3
MODEL. Strength is the slider scale: `0` is off, `±1` is the trained unit on
files whose sidecar says `unit_scale: 1.0`.

## Converter

Do not add a second converter here. Convert native LoRANetwork files with
[`scripts/convert_lora_comfyui.py`](https://github.com/mikkel/conceptmod/blob/main/scripts/convert_lora_comfyui.py)
on [mikkel/conceptmod](https://github.com/mikkel/conceptmod) at
[`8f865fe`](https://github.com/mikkel/conceptmod/commit/8f865fea59e02d439a479d80466196044ed00076)
or later. Music 3 backends (`music3`, `music3_lm`) landed in that commit.
Detection is from `lora_unet-` / `lora_te-` keys.

```bash
python /path/to/conceptmod/scripts/convert_lora_comfyui.py path/to/adapter.safetensors
```

The published recordings were rendered with the native adapters. ComfyUI
precision and sampler settings can change the audio.
