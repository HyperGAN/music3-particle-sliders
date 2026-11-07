# Installation cohorts

The root0.4.0 package owns native Music3 code and selects model-independent core0.14.0 at
`8a696eb33c1a9d34472afad3a3199bd4776a797f` and published PG0.9.5.
Use a fresh isolated environment and `python -m pip install '.[response-fixture,test]'`.
The common response lifecycle and native-only reader retain their API.

| Cohort | Route | Scope |
|---|---|---|
| Current core0.14 / PG0.9.5 | Root 0.4 product-owned native package, release-0.9.5/auto | Narrow random CPU fixtures |
| Retained core0.13 / PG0.9.5 | [Original 0.3 source cohort](cohorts/pg095-response-3dd-20261105/README.md) | Original packets, captures and measured results only |
| New product-owned native continuous PG0.9.5 | Explicit release-0.9.5/continuous plus strict BF16 eager policy | Fresh native qualification required for this source cohort |
| Historical core5d / PG094 and PG801 | Original source/environment; preserved response/native guides | Earlier packets and measured facts only |
| Previous coreddb PG094 | cohorts/pg094-response-ddb-20261104 | Independently buildable archive, unchanged |
| Historical core0.8.2/core0.6 PG091 | cohorts/legacy-response and original CLI locks | Separate original environments |
| HNDL/core0.12 candidate | cohorts/hndl | Separate pin, unchanged |

No old checkpoint/source identity is rewritten. Existing ordinary LoRA controls,
ComfyUI assets and historical CLI scripts keep their recipes and load paths.
Read [current usage](docs/response-session-proposal.md),
[native settings](NATIVE-CONTINUOUS.md), and the preserved
[PG094 guide](docs/response-session-pg094.md). The immutable Git dependency is
intentional; these direct-VCS wheels are not PyPI uploads.
