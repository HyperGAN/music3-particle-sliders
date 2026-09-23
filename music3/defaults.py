"""Pinned Music 3 ids. Importing this module hits no network."""

from __future__ import annotations

CORE_COMMIT = "a119ca1ecd3d5d6c437065839d22739b04f2f4d8"
CORE_URL = (
    "particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git@"
    f"{CORE_COMMIT}#subdirectory=packages/particle-sliders-core"
)

BASE_MODEL_ID = "MiniMaxAI/MiniMax-Music-3"
HUB_WEIGHTS_ID = "ntc-ai/minimax-music3-concept-sliders"
HUB_SPACE_ID = "ntc-ai/minimax-music3-concept-sliders"
HUB_WEIGHTS_URL = f"https://huggingface.co/{HUB_WEIGHTS_ID}"
HUB_SPACE_URL = f"https://huggingface.co/spaces/{HUB_SPACE_ID}"
RELEASE_TAG = "uni16-fresh-selected-v2"
RELEASE_POLICY = "quality-later-v2"

# Sibling project. Not a Music 3 weight id.
YUE2_WEIGHTS_ID = "ntc-ai/yue2-concept-sliders"
YUE2_SPACE_ID = "ntc-ai/yue2-concept-sliders"

PRODUCT_NAME = "music3-particle-sliders"
CONCEPTMOD_REPO = "https://github.com/HyperGAN/conceptmod"
PARTICLE_SLIDERS_REPO = "https://github.com/HyperGAN/particle-sliders"
ANIMA_PRODUCT = "https://github.com/HyperGAN/anima-particle-sliders"
KREA2_PRODUCT = "https://github.com/HyperGAN/krea2-particle-sliders"

# Comfy conversion stays upstream. This repo does not vendor a second converter.
COMFY_CONVERTER_REPO = "https://github.com/mikkel/conceptmod"
COMFY_CONVERTER_COMMIT = "8f865fea59e02d439a479d80466196044ed00076"
COMFY_CONVERTER_SCRIPT = "scripts/convert_lora_comfyui.py"
COMFY_LM_KEY_EXAMPLE = "text_encoders.model.layers.N.self_attn.q_proj.lora_A.weight"
COMFY_TF_KEY_EXAMPLE = (
    "diffusion_model.diffusion_transformer.transformer.layers.N.self_attn.to_qkv.lora_A.weight"
)

# Published September 16, 2026 listen card.
RELEASE_SEED = 1709
RELEASE_SCALES = (0.0, 1.0)
LISTEN_DURATION_S = 20.0
# Historical energy wav demo in conceptmod/textsliders/infer_music3.py.
ENERGY_DEMO_SCALES = (-2.0, -1.0, 0.0, 1.0, 2.0)
ENERGY_DEMO_DURATION_S = 6.0

# Stamp-locked adapter size. Exposed here so CLIs can reject a fork early.
ADAPTER_RANK = 8
ADAPTER_ALPHA = 8.0

DEFAULT_PROMPTS = "configs/music3/prompts-particle.yaml"
ENERGY_PROMPTS = "configs/music3/prompts-music3.yaml"
TRANSFORMER_CONFIG = "configs/music3/config-music3.yaml"
ENCODER_CONFIG = "configs/music3/config-music3-encoder.yaml"
SAMPLE_CARDS = "configs/music3/sample-cards.json"

# Music particle bridge used adv_batch 64. That key is a model surface.
MUSIC_ADV_BATCH = 64
DUMMY_HIDDEN = 16
DUMMY_MAX_STEPS = 4
