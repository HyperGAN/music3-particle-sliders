# Installation cohorts

The root package is the current **core0.12.0 / official ParticleGAN0.9.4**
numerical-response cohort. Its immutable shared source is
`5d68473a098801ee1ab3ebbb12463daef51c6346`; use Python 3.11+ and
`python -m pip install '.[response-fixture,test]'` in a new isolated environment.
`requirements.txt` delegates to this same package pin. Current training and
inference use `particle-slider --interface response` as documented in
[the current response guide](docs/response-session-proposal.md).

| Cohort | Entry point / lock | Status |
|---|---|---|
| Current core0.12.0 / official PG0.9.4 | Installed response/inference registry; `music3/response-core.lock.json` | Narrow CPU fixtures; default wide vectors refused |
| Explicit native core5d / PG801 continuous source | Same registry with explicit profile/domain/strict BF16 eager settings; `NATIVE-CONTINUOUS.md` | CPU source/assets ready; full native GPU/resume/audio qualification pending |
| Previous coreddb / official PG0.9.4 | Independently buildable `cohorts/pg094-response-ddb-20261104/` | Exact old package bytes and SHA manifest preserved |
| Historical core0.8.2 / PG0.9.1 | Product source `ed1c667` before this upgrade; archived lock and guide in `cohorts/legacy-response/` | Reproduce in a separate checkout and environment |
| Historical core0.6.0 / PG0.9.1 | `scripts/train_*.py`, `scripts/infer_music3.py`, `core.lock.json`, `requirements-legacy.txt` | Original locked CLI only; tests skip in other core environments |
| Independent HNDL/core0.12 candidate | `cohorts/hndl/pyproject.toml` and its own source lock | Separate environment; unchanged by this upgrade |

Never combine these environments or rewrite old checkpoint/source identity.
The root's historical CLI scripts retain their original implementation and
core0.6.0 lock; they are not the current training command. The archived response
lock retains the exact core0.8.2 bytes. Published ordinary native LoRAs and
ComfyUI recordings preserve their original model identities and load path.
No existing weights are retrained, converted or republished by this change.

The direct VCS dependency is intentional for immutable source installation;
these wheels must not be uploaded to PyPI. A successful package installation
is not native audio qualification. Official width>=170 flat AR response and
media rendering gates remain enforced.
