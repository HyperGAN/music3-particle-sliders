# Product-owned Music3 adapters

The root 0.4 cohort owns native Music3 code in this repository. `music3.native`
owns prompt packing, the routed adapter format and host attachment;
`music3.response_adapter` selects the Music3 final-token hidden coordinates.
`music3.lifecycle` owns model/tokenizer admission, paired teachers, native
observation and resource cleanup. Distillation consumes these same owned types.

The dependency is the model-independent core 0.14. The common package supplies
the Conditional ParticleGAN game, official optimizer integration, checkpoint
and response lifecycles, learning-rate controls, generic AR binding, LoRA,
routing tensors and transactional attachment helpers. No GAN implementation or
optimizer equation is copied into the product. Native model imports remain
lazy, and the default CPU fixture uses a small random Qwen3 configuration.

Prospective configuration requires `ema_decay=None`; exports contain raw current
weights. Native adapter loading refuses known averaged provenance before
inspecting the host. Scalar optimizer history stays upstream and unchanged.
Existing native adapter format names and tensor keys remain available, while
full resume and inference packets bind the new product/core source identities.

This is a new opt-in source cohort. The previous 0.3/core-0.13 pin and original
source receipts are retained in
[`cohorts/pg095-response-3dd-20261105`](../cohorts/pg095-response-3dd-20261105/README.md).
Earlier CLI, response and HNDL cohorts and measured native captures retain their
original immutable sources and environments. Existing jobs are not retargeted,
and old packets are not relabeled as belonging to core 0.14.

CPU tests establish wiring, owned gradients, raw export/load, rollback, exact
continuation and public response replay. They do not qualify pretrained Music3
updates, device memory, acoustic rendering or old measured learning rates for
this new source cohort. Historical native measurements remain facts about the
original 0.3 cohort.
