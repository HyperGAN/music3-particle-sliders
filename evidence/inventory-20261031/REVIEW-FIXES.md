# Published-source review follow-up

Reviewed base: `e56d3d15d974bdac79872ba6af53e016d52e4463` (draft PR7).
No core, default runtime pin, historical lock or checkpoint identity was changed.

1. **Cleanup skipping the slider / masking a primary error — fixed.**
   `_cleanup_parts` uses `try/finally` to attempt shared checkpoint and native
   slider cleanup independently. Normal close raises the first cleanup error
   with every failure retained in `cleanup_errors`. Factory/context failure
   preserves the exact original exception and attaches cleanup errors plus a
   reachable `retry_cleanup` callback. Slider cleanup attempts sibling forwards
   and native flag restoration even if one owner conflicts. Only successful
   ledger entries are removed; foreign forwards are preserved for explicit retry.
   Regression tests cover failing checkpoint close with actual shared handles,
   both owners failing during a context-body error, and critic construction
   failing while checkpoint cleanup also fails. A two-forward conflict test
   verifies nested cleanup errors and both retry ledgers remain reachable.
   The primary error is unchanged.

2. **Partial forward-patch failure — fixed.** The shared projection is built
   with `attach=False`; the product records the saved/attached forward before
   assignment. Setters failing before mutation or after mutation both restore
   all previous patches, modes, flags and frozen bytes. The existing graph-build
   failure regression remains. See both parametrizations of
   `test_partial_forward_assignment_failure_rolls_back_all_patches`.

3. **Initial scale0 outside a context — verified; initial behavior needed no
   change.** Shared HNDL projections already start with `multiplier=0.` and the
   product starts at scale0. The new explicit regression sets every up matrix
   nonzero, compares the direct native forward against the pre-attachment base,
   proves scale1 changes the result, and proves the direct forward is exact base
   again after leaving the strength context.

4. **Delegated source validation — clarified and strengthened.** Product
   `verify_core` validates the canonical complete core source map, exact wheel
   or VCS provenance, and import origins. Shared `fibo_hndl.verified_builder`
   validates HNDL0.10, all HNDL Python bytes, the six HyperGAN builder files and
   clean build provenance for5f368a. The factory now calls it before allocating
   Qwen3. Two fault-injection tests alter the computed HNDL/HyperGAN source hash
   and verify refusal before model allocation. Installed files are not modified.

5. **PyPI publication — explicitly prohibited by these manifests.** Both
   artifacts contain direct VCS dependencies and are unsuitable for PyPI upload.
   Documentation now says this explicitly. No publication or upload is performed.

6. **Package-data globs — verified against actual outputs; no glob defect.**
   `scripts/check_inventory_artifacts.py` opens both wheels and sdists and checks
   exact bytes for stable `response-core.lock.json`/`data/sample-cards.json` and
   candidate `core.lock.json`/`critic.hndl`. It also checks all15 stable and5
   candidate Python/data/graph files in both wheels and installed packages against
   source, and confirms direct-VCS
   metadata. Output: `review/packaging.json`.

7. **Reproducibility — replaced incomplete commands.** The checked-in offline
   script creates fresh virtual environments; selects read-only CPU dependency
   binaries and records their versions/origins; hash-checks and explicitly
   installs exact core0.8.2 and0.12 wheels plus PG/HNDL; builds HyperGAN from a clean
   detached5f368a checkout; builds/installs both products; then tests repository
   paths using `python -I` from `/tmp`. No copied test suites are used. Exact
   commands, input paths, wheel hashes and configurable source runtimes are in
   `commands.txt` and the script. Local owner-provided wheels are the recorded
   qualification inputs. An exact public Git fetch of core4abc0eb now also passes;
   see `review/core-public-source-receipt.json`. Its public builder source was
   fetched and verified byte-for-byte against the qualified core source.

Completed fresh-environment results: **37 retained-cohort tests,24 HNDL tests,
7 source/config tests passed**, both dependency checks pass, both wheel/sdist
builds and package-data checks pass. `review/` contains the literal output.
Torch2.14.1+cpu/Python3.12.15 are reused read-only; no pretrained/GPU qualification
or Python matrix is claimed. The build source is clean HyperGAN5f368a with
`dirty:false` provenance. No GPU, model weights, package upload, merge or release.

Remaining gates: native width4096 refusal, pretrained/tokenizer/audio/listening
qualification and native distillation/ComfyUI. The previous exact-core source
publication blocker is resolved. The candidate's source-validation hashes keep
checkpoint cohorts distinct; historical environments remain intact.
