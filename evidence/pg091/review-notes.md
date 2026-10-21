# Final product review notes

The source-free product uses proposed shared package
`9be81c1ccfc8ad87eac3efa93fd5a16d0d508bb5` (local wheel SHA-256
`60629b05298559340807084d278bb256781bf0421c6d928c23e57199e2b37dc8`).
The source, helper and math implementations are installed from that wheel;
this repository contains CLI/provenance wiring and consumer tests.

The product review found and repaired:

- A stale `load_game_class()` wrapper still passed removed checkout arguments.
  Its default now returns the verified installed `ConditionalPG09`; supplied
  historical checkout paths fail explicitly.
- Inference omitted its Music3 family binding when calling the shared probe
  CLI. It now supplies `--family music3`, and a wrong-family export rejects
  before fixture allocation or output creation.
- The installed-file lock compared the build-generated `WHEEL` Generator
  line byte for byte, while a future exact Git build may select another
  allowed setuptools version. Only verified immutable public Git provenance
  now permits that header's value to vary. Its position and every other
  WHEEL byte stay bound to the exact reviewed wheel. All source, METADATA,
  archive and module-origin checks remain strict. Tests cover the accepted
  variation and atomic refusal of foreign/branch/local provenance, malformed
  metadata, altered other headers, source or dependency metadata.

A second read-only review found no remaining concrete import, CLI, lineage
or metadata-exception counterexample. This is consumer review, not Ada
approval. The final CPU XML and source hashes are in the acceptance receipt.

The package owner's publication was denied and did not execute. Neither
these fixture tests nor the metadata exception establishes a successful
public Git fetch/build. That gate remains pending exact publication
approval, followed by actual immutable VCS installation acceptance.

Malformed shared optimizer history, CDF clocks, native export loading,
prior metrics history and factory RNG were repaired centrally by the
shared owners. The product carries their exact final local wheel and
consumer regressions; it does not duplicate those implementations.
