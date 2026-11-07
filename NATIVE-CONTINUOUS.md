> Retained 0.3/core-0.13 source evidence: the measurements and pins below belong
> to the original product/core cohort. The root 0.4 product-owned adapter and
> core-0.14 integration require a fresh source/environment cohort and native
> qualification. See [ownership and migration](docs/product-owned-native.md).

# Native continuous Music3 — published PG0.9.5

## Completed native response measurement

The source-qualified 2026-11-05 cohort completed two uninterrupted native updates versus independent one-plus-one continuation, exact full-state continuation, RAW export/public response/OFF replay, and a 128-update LR nominee on actual pretrained 4,096-wide BF16 geometry. Its [measured combination and exact preset](LEARNING-RATES.md#measured-native-pg095-combination) are G=1e-6, D=0.018000000000000002, P=0.029999999999999995. The unused two-example final RAW score was 0.8928904545147202; development at 128 was 1.0382026088532337. The result is best tested at the declared lower G bound, unbracketed, with fixed D/P. Defaults are unchanged. No audio generation or listening qualification is claimed.

The [sanitized receipt](evidence/pg095-native-lr/summary.json) supersedes the earlier pending-native status below for this exact source/profile. Earlier CPU and failed-admission observations remain historical evidence.

## Native setup and earlier readiness evidence

The default narrow CPU fixture uses `release-0.9.5/auto`. Native4096-wide
responses require `pg_source_profile="release-0.9.5"`,
`penalty_input_domain="continuous"`,
`native_numerical_profile="music3-bf16-eager-strict-pg095-v1"`,
`device="cuda:0"` and `base_precision="bfloat16"`.
Core0.13.0 is pinned to `3dd672a7f9b8c169daef90fd63e4bb42b9ddc423`;
the published PG0.9.5 wheel and full577-file shared source map are verified.
The old source801 profile refuses. Install in a fresh environment with the
root package's exact dependencies; no separate PG801 overlay is needed.

Native dependencies are actual Torch`2.9.1+cu128` / CUDA12.8, Transformers 5.15.0, Diffusers `d5baa4fb548294f47dbca49890abd4b291204c60`, huggingface-hub 1.27.0, safetensors 0.8.0, numpy 2.2.6, and tokenizers 0.22.2. The `native-source` extra installs these runtime dependencies; select the CUDA12.8 Torch build explicitly in a fresh PG0.9.5 environment. Installing that extra alone does not change the default ParticleGAN source profile. The pinned Diffusers caption/lyrics packing and genuine Qwen3 eager attention are used. The asset lock binds all LM/tokenizer files for MiniMax-Music3 revision`bd348f9c49ea3c1b39f33ace3436f8fad435f24e` plus exact owned Transformer/Diffusers source files. Supply local `model_dir`, `manifest`, and `manifest_sha256`; no download occurs. Complete 399 BF16 checkpoint tensors, 8,584,475,648 elements, are compared before adapter attachment.

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

The PG0.9.5 protocol is `music3-native-final-ar-hidden-v2-pg095`.
Model geometry, dtype, numerical flags and inherited rate defaults are unchanged.
This is a new source cohort: no old full-state or portable packet is rebound.
Earlier source801 import/weight/probe observations remain historical, as
recorded in [the preserved native guide](docs/native-pg801.md). They do not
qualify PG0.9.5 native training or learning rates; the fresh PG0.9.5 evidence is linked above.
