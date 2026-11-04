# Historical PG0.9.1 numerical-response cohort

This directory preserves the original `response-core.lock.json` and source
proposal byte for byte. They are historical evidence, not the active root lock.
For exact legacy runtime reproduction, use product commit
`ed1c6675b526116f9cff5d3ed20f3a8ee546e3fb` in a
separate checkout and environment with core0.8.2 at
`f6807a91546e43e198645a3f8925d2dc7c9e03b4` / official ParticleGAN0.9.1.
Do not load its training packets in the new PG0.9.4 response cohort.
The even older core0.6.0 CLI retains `core.lock.json` at the repository root.
