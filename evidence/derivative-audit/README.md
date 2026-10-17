# Music3 derivative audit receipts

The base is HyperGAN/music3-particle-sliders main at b40d2aa6fed745eda3043d3d3b11e362ec2f8dfe. Work is isolated on audit/music3-shared-pg09-20261017. Existing audio9262, core3e94 and pinned product b40d2aa remain clean and unchanged.

`audit-receipt.json` records exact commands, native interpreter/package origins, 49 PG source hashes, tested product-source hashes, all six entry-point outcomes, fixes and remaining gates. `product-tests.xml` records 33 passing product tests. `shared-native-width-tests.xml` records four passing shared-domain tests at native width4096 using random Qwen3 and the reviewed ALZ profile. These run on CPU; no pretrained tensor or acoustic model is loaded.

`native-plan.json` is a non-dispatched proposal against staged revision bd348f9c49ea3c1b39f33ace3436f8fad435f24e. Its environment contains exact source roots and the separately reviewed overlay; it has no real GPU assignment, admission ID or guardian. `native-plan-imports.json` confirms the exact original core/audio source hashes and width4096 continuous-domain validation in a fresh native interpreter process. It also confirms CUDA was not initialized. The proposed output directory is absent.

Independent product review identified two native-plan wiring errors (missing fresh-process import roots and invalid default native output root); both were repaired with regression tests. Final review found no remaining concrete bug. Ada review of the exact published head remains pending and is required before parent-controlled merging.

The existing public PR #3 is a separate historical FormulationGame draft at 2f1ff5a952b2c3e81365ba2aeeeb5edd94fcd76b. This PG0.9 draft targets current main and does not adopt that unmerged code. No source-bound checkpoint is migrated to the new package namespace. Task14 owns that separate extraction and its review.
