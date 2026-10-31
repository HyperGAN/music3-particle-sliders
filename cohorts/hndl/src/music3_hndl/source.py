"""Exact candidate admission before importing the core or native libraries."""
import hashlib
from importlib import metadata, util
import json
from pathlib import Path
import sys

CORE_COMMIT = "a794e3b4a8813b012f2c4faa3da82b78d4a90214"
LOCK_SHA256 = "9c354ffa98523fa9ed22f3936aea8922a120e530ce171db4ba8c7ed2e2707887"


def verify_core():
    contents = Path(__file__).with_name("core.lock.json").read_bytes()
    if hashlib.sha256(contents).hexdigest() != LOCK_SHA256:
        raise RuntimeError('Music3 HNDL canonical source lock bytes differ')
    lock = json.loads(contents)
    if lock['commit'] != CORE_COMMIT:
        raise RuntimeError('Music3 HNDL candidate source lock contradicts its cohort')
    dist = metadata.distribution('particle-sliders-core')
    if dist.version != '0.12.0' or metadata.version('particlegan') != '0.9.4':
        raise RuntimeError('Music3 HNDL requires its separate exact core 0.12/PG 0.9.4 environment')
    registered = {str(name) for name in dist.files or ()}
    expected = lock['sources']
    source_names = {name for name in registered if name.startswith(('particle_sliders/', 'concept_slider_core/'))
                    and '__pycache__' not in Path(name).parts}
    if source_names != set(expected):
        raise RuntimeError('Music3 HNDL candidate installed source inventory differs')
    root = Path(dist.locate_file('')).resolve()
    paths = {}
    for name, digest in expected.items():
        path = Path(dist.locate_file(name)).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Music3 HNDL candidate installed source differs: ' + name)
        paths[name] = path
    provenance = json.loads(dist.read_text('direct_url.json') or '{}')
    if 'vcs_info' in provenance:
        vcs = provenance['vcs_info']
        if (provenance.get('url', '').rstrip('/').removesuffix('.git') != 'https://github.com/HyperGAN/particle-sliders'
                or provenance.get('subdirectory') != 'packages/particle-sliders-core'
                or vcs != dict(vcs='git', requested_revision=CORE_COMMIT, commit_id=CORE_COMMIT)):
            raise RuntimeError('Music3 HNDL candidate VCS provenance differs')
    elif provenance.get('archive_info', {}).get('hashes', {}).get('sha256') != lock['wheel_sha256']:
        raise RuntimeError('Music3 HNDL candidate requires its exact reviewed wheel or VCS provenance')
    for name, module in tuple(sys.modules.items()):
        if name.split('.')[0] not in ('particle_sliders', 'concept_slider_core'):
            continue
        relative = name.replace('.', '/')
        expected_path = paths.get(relative + '.py', paths.get(relative + '/__init__.py'))
        if expected_path is None or Path(getattr(module, '__file__', '')).resolve() != expected_path:
            raise RuntimeError('foreign shared module already imported: ' + name)
        if hasattr(module, '__path__') and tuple(Path(p).resolve() for p in module.__path__) != (expected_path.parent,):
            raise RuntimeError('foreign shared package search path: ' + name)
    spec = util.find_spec('particle_sliders')
    if (spec is None or Path(spec.origin or '').resolve() != paths['particle_sliders/__init__.py']
            or tuple(Path(p).resolve() for p in spec.submodule_search_locations or ())
               != (paths['particle_sliders/__init__.py'].parent,)):
        raise RuntimeError('foreign shared package import origin')
    return lock
