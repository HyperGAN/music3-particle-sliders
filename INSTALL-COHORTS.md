# Installation cohorts

The root0.3.0 package selects core0.13.0 at
`3dd672a7f9b8c169daef90fd63e4bb42b9ddc423` and published PG0.9.5.
Use a fresh isolated environment and `python -m pip install '.[response-fixture,test]'`.
The common response lifecycle and native-only reader retain their API.

| Cohort | Route | Scope |
|---|---|---|
| Current core0.13 / PG0.9.5 | Root package, release-0.9.5/auto | Narrow random CPU fixtures |
| Current native continuous PG0.9.5 | Explicit release-0.9.5/continuous plus strict BF16 eager policy | CPU software/source admission; fresh native GPU proof pending |
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
