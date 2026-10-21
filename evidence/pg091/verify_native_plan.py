"""Hash actual staged native inputs through the public plan-only API."""
import importlib.abc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
PROFILE = Path('/ml2/codex-storage/future-workspaces/2026-10-17/task-14/pg091-release/continuous')
sys.path[:0] = [str(PROFILE), str(ROOT)]
class NoHistoricalSource(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'conceptmod','analysis','qualify_audio_pg09_live','pg09_audio_cpu_fixtures'}:
            raise AssertionError('historical source import: '+fullname)
sys.meta_path.insert(0, NoHistoricalSource())
import torch
from music3.shared import bootstrap
from music3.train import parse_args, train
shared = bootstrap()
def forbidden(*args, **kwargs):
    raise AssertionError('native planning loaded a model or dispatched training')
shared.native.Music3Backend = forbidden
shared.cli.make_audio_game = forbidden
out = Path('/mnt/nvme/sliders/qualification/runs/music3-packaged-pg091-unadmitted')
assert not out.exists() and not torch.cuda.is_initialized()
plan = train(parse_args([
    '--live', '--model_dir', '/mnt/nvme/sliders/qualification/models/music3-bd348f9',
    '--manifest', '/mnt/nvme/sliders/qualification/models/music3-bd348f9/qualification-manifest.json',
    '--prompts_file', '/mnt/nvme/sliders/audio-adapters-pg09-20261017/conceptmod/textsliders/data/prompts-yue2-metal-arm-b.yaml',
    '--heldout_prompts', '/mnt/nvme/sliders/audio-adapters-pg09-20261017/conceptmod/textsliders/data/prompts-yue2-metal-arm-b-eval.yaml',
    '--save_dir', str(out), '--pg_source_profile', 'alz-continuous-v1', '--penalty_input_domain', 'continuous',
]))
assert not plan['dispatch_allowed'] and plan['output_absent'] and not out.exists()
assert not plan['native_audio_rendered'] and not torch.cuda.is_initialized()
plan.update(source_receipt=shared.receipt, profile_root=str(PROFILE),
            source_free_imports=True, models_constructed=False, tensors_deserialized=False,
            cuda_initialized=False, admission_granted=False)
(ROOT/'evidence/pg091/native-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print('Actual staged LM/tokenizer hashes verified; plan only, no model/tensor/CUDA/dispatch/output.')
