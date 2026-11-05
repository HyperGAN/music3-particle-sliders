# Native continuous Music3 source candidate

The default narrow CPU fixture retains official ParticleGAN 0.9.4 `release-0.9.4` / `auto`. The genuine Music3 4096-wide hidden response is separately opt-in: `pg_source_profile="source-801ae1e-continuous-v1"`, `penalty_input_domain="continuous"`, `native_numerical_profile="music3-bf16-eager-strict-v1"`, `device="cuda:0"`, and `base_precision="bfloat16"`. The default release continues to refuse wide vectors. There is no projection, source fallback or acoustic rendering API.

The candidate uses core `5d68473a098801ee1ab3ebbb12463daef51c6346` and the distinct exact ParticleGAN source `801ae1ead8f88b88eca8f5398137e3e1bc01ad36`. The latter reports the same 0.9.4 version as the official wheel but has different source bytes; install it only in a separate candidate environment. Product source admission verifies the shared source closure and exact VCS origin.

Native dependencies are actual Torch`2.9.1+cu128` / CUDA12.8, Transformers 5.15.0, Diffusers `d5baa4fb548294f47dbca49890abd4b291204c60`, huggingface-hub 1.27.0, safetensors 0.8.0, numpy 2.2.6, and tokenizers 0.22.2. The `native-source` extra installs these runtime dependencies; select the CUDA12.8 Torch build explicitly and install the distinct qualified PG801 source in its own environment. Installing that extra alone does not change the default ParticleGAN source profile. The pinned Diffusers caption/lyrics packing and genuine Qwen3 eager attention are used. The asset lock binds all LM/tokenizer files for MiniMax-Music3 revision`bd348f9c49ea3c1b39f33ace3436f8fad435f24e` plus exact owned Transformer/Diffusers source files. Supply local `model_dir`, `manifest`, and `manifest_sha256`; no download occurs. Complete 399 BF16 checkpoint tensors, 8,584,475,648 elements, are compared before adapter attachment.

The caller owns the numerical settings. Before preparation, configure deterministic algorithms with warn-only disabled; `CUBLAS_WORKSPACE_CONFIG=:4096:8`; highest FP32 precision; cuDNN deterministic, benchmark disabled; both TF32 flags and FP16/BF16 reduced-precision reductions disabled; flash, memory-efficient and cuDNN SDPA disabled, math SDPA enabled with low-precision math reduction disabled. Forced TF32 environment overrides are refused. The product records actual settings and explicit eager attention, refuses changes before responses and observation, and never repairs flags implicitly.

The entire native prepare, including provider, critic, prior, streams and optimizers, seeds CPU plus only the requested CUDA generator inside a restoring fork scope. Caller RNG restores on success and failure. Native inference uses the same owned scope and creates no training provider, critic or optimizer. Standard `lr`, `d_lr_mult`, `prior_lr_mult` retain the shared recipe/defaults and exact actual optimizer values. No native learning rate is recommended by CPU evidence.

CPU checks cover strict-policy refusals, three-visible-device RNG mocks, actual small Qwen3/provider/critic/prior state equality across fresh processes and caller seeds, and exact two-update versus one-plus-one continuation. These checks and full-class META/asset hashing are source/readiness evidence only: the full pretrained model has not yet completed a GPU update, exact native resume, memory measurement, audio generation, preservation evaluation, or listening review. The BF16 language-model weight floor alone is 15.9898 GiB; a future GPU proof needs separately admitted headroom.

The previous default core `ddb41423bcefc1fe76cd94d0f9b367353b3ea137` response package is preserved byte-for-byte and independently buildable in `cohorts/pg094-response-ddb-20261104`, with SHA manifest beside it. Older CLI, response and HNDL cohorts retain their bytes and environments; their packets are not relabeled or migrated.


Version0.2.1 corrects strict import admission for the exact virtual legacy aliases
created by the pinned Transformers5.15 initializer. The real cached Music3
AutoTokenizer factory returns native Qwen2Tokenizer with audio-start151669;
it also initializes215 source-defined aliases whose __file__ is intentionally
None. The previous full attempt loaded and verified399 BF16 base tensors, then
refused at the old import-origin gate before any teacher cache or update. That
failed source/environment/output remains separate and unchanged.

Admission now derives only the three explicit aliases and212 model-image aliases
from the locked runtime source inventory, verifies the exact initializer bytes,
registered module identity, exact delegation function code and closure target,
and the locked canonical target source bytes. Unknown originless modules,
replaced aliases, changed delegations or source hashes still refuse. The runtime
dependencies, numerical flags, loaded weights, PG/core source and GAN math are
unchanged. Actual checkpoint tokenizer plus full native source admission passes
on CPU with CUDA hidden; this does not qualify a pretrained update, native resume,
portable response, or acoustic quality. Any new native attempt requires a fresh
source/environment/capture cohort and explicit resource admission.


LR screening and confirmation use one ranking criterion: RAW held-out
teacher/student relative SSE, pooling positions within each example and then
taking the equal-example mean. Preservation drift, late/weak breakdowns,
acoustic rendering, listening and other diagnostics do not gate selection.
Correct source/asset identity, native updates, exact resume and portable response
remain correctness prerequisites. No acoustic baseline or listening task is
required before the native hidden-response LR screen.
