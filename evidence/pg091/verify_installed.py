"""Read-only installed API/legacy-lineage proof; no updates or native loading."""
from copy import deepcopy
import hashlib
from importlib import metadata
import importlib.abc
import json
import inspect
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

class NoHistoricalSource(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'conceptmod', 'analysis', 'pg09_audio_cpu_fixtures', 'qualify_audio_pg09_live', 'run_audio_pg09_smoke', 'inspect_audio_pg09_assets'}:
            raise AssertionError('historical source import: ' + fullname)
sys.meta_path.insert(0, NoHistoricalSource())

import torch
import transformers
import particlegan
from music3.shared import bootstrap


def exact(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a == b


def main():
    shared = bootstrap()
    run = shared.cli.make_audio_game('music3')
    before = deepcopy(run.game.state_dict())
    legacy = Path('/tmp/music3-first-probe-20261017')
    evidence = {}
    for filename, load in (('checkpoint.pt', lambda value: run.game.load_state_dict(value['game'])),
                           ('adapter-export.pt', run.game.load_export_state_dict)):
        path = legacy / filename
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        value = torch.load(path, weights_only=True, map_location='cpu')
        state = torch.get_rng_state().clone()
        try:
            load(value)
        except ValueError as error:
            reason = str(error)
        else:
            raise AssertionError('old PG0.9.0 artifact was accepted')
        assert exact(before, run.game.state_dict())
        assert torch.equal(state, torch.get_rng_state())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        evidence[filename] = dict(path=str(path), sha256=digest, rejected=True,
                                  state_and_rng_unchanged=True, bytes_unchanged=True, reason=reason)
    pg_root = Path(particlegan.__file__).resolve().parent
    pg_sources = {str(p.relative_to(pg_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(pg_root.rglob('*.py'))}
    assert len(pg_sources) == 49 and not torch.cuda.is_initialized()
    receipt = dict(status='INSTALLED_PACKAGE_AND_LEGACY_REJECTION_PASSED',
        source_free=True, new_lineage=shared.lock['lineage'], source_receipt=shared.receipt,
        runtime=dict(executable=sys.executable, torch_version=str(torch.__version__), torch_file=torch.__file__,
                     transformers_version=transformers.__version__, transformers_file=transformers.__file__,
                     particlegan_version=metadata.version('particlegan'), particlegan_file=particlegan.__file__,
                     pg_sources=pg_sources, cuda_initialized=torch.cuda.is_initialized()),
        fixture=dict(model_class=type(run.model).__module__+'.'+type(run.model).__name__,
                     hidden_width=run.model.config.hidden_size, vocab_size=run.model.config.vocab_size,
                     audio_start_tokens=[p.row.neutral_ids[-1] for p in run.rows],
                     native_model_source=dict(path=inspect.getsourcefile(type(run.model)), sha256=hashlib.sha256(Path(inspect.getsourcefile(type(run.model))).read_bytes()).hexdigest())),
        game_signature={key:run.game.state_dict()["config"][key] for key in ("pg_version","pg_release_wheel_sha256","pg_sources_sha256","implementation_sources","pg_source_profile","penalty_input_domain","resolved_penalty_domain","response_shape","conditions_sha256","rows_sha256","sampling","objective","generator_arm","recipe","backend")},
        legacy_artifacts=evidence, pretrained_weights_loaded=False, audio_rendered=False)
    destination = ROOT / 'evidence/pg091/installed-api-receipt.json'
    destination.write_text(json.dumps(receipt, indent=2)+'\n')
    print('Installed sources verified; actual old checkpoint/export refused without state/RNG/byte changes; no CUDA.')

if __name__ == '__main__':
    main()
