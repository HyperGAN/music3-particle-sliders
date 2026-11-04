"""Actual pinned tokenizer compatibility; no pretrained model/CUDA allocation."""
from importlib import metadata
from pathlib import Path
import json,sys,types
import pytest
import torch
from music3 import native_assets as assets
from test_native_profile import strict_math

@pytest.fixture
def aliases():
    pytest.importorskip('transformers')
    if metadata.version('transformers')!='5.15.0':pytest.skip('exact native Transformers5.15 environment required')
    directory=Path(metadata.distribution('transformers').locate_file('transformers')).resolve()
    lock=json.loads(assets.LOCK.read_text())['runtime_sources_sha256']
    permitted=assets._transformers_aliases(directory,lock)
    return directory,lock,permitted


def check(name,module,binding):
    return assets._verify_transformers_alias(name,module,*binding)


def test_actual_initializer_all_215_aliases_have_exact_code_target_and_source(aliases):
    expected=aliases[2][0]
    assert len(expected)==215
    actual={name:module for name,module in tuple(sys.modules.items())
        if name.split('.')[0]=='transformers' and vars(module).get('__file__') is None}
    assert set(actual)==set(expected)
    for name,module in actual.items():assert check(name,module,aliases)==expected[name]
    assert not torch.cuda.is_initialized()


def test_unknown_originless_module_is_refused(aliases,monkeypatch):
    name='transformers.unapproved_virtual_module';module=types.ModuleType(name)
    monkeypatch.setitem(sys.modules,name,module)
    with pytest.raises(ValueError,match='unknown originless'):check(name,module,aliases)


def test_same_name_replaced_module_is_refused(aliases,monkeypatch):
    name='transformers.tokenization_utils_fast';original=sys.modules[name]
    replacement=types.ModuleType(name);replacement.__dict__.update(vars(original))
    monkeypatch.setitem(sys.modules,name,replacement)
    with pytest.raises(ValueError,match='registered alias'):check(name,replacement,aliases)


@pytest.mark.parametrize('field',['__getattr__','__dir__','__doc__','__file__','__package__','extra_attribute'])
def test_modified_real_alias_is_refused(aliases,monkeypatch,field):
    name='transformers.tokenization_utils_fast';module=sys.modules[name]
    value=(lambda *args:None) if field in ('__getattr__','__dir__') else 'changed'
    monkeypatch.setattr(module,field,value,raising=False)
    with pytest.raises(ValueError,match='alias'):check(name,module,aliases)


def test_wrong_delegation_target_closure_is_refused(aliases,monkeypatch):
    name='transformers.tokenization_utils_fast';module=sys.modules[name]
    getter=vars(module)['__getattr__'];target_function=getter.__closure__[0].cell_contents
    cell=target_function.__closure__[0];old=cell.cell_contents
    try:
        cell.cell_contents='.tokenization_utils_base'
        with pytest.raises(ValueError,match='delegation'):check(name,module,aliases)
    finally:cell.cell_contents=old


@pytest.mark.parametrize('which',['initializer','target'])
def test_locked_initializer_or_target_source_digest_cannot_be_relabelled(aliases,which):
    directory,lock,permitted=aliases;changed=dict(lock)
    key='transformers:transformers/'+('__init__.py' if which=='initializer' else 'tokenization_utils_tokenizers.py')
    changed[key]='0'*64
    with pytest.raises(ValueError,match='source differs'):
        if which=='initializer':assets._transformers_aliases(directory,changed)
        else:check('transformers.tokenization_utils_fast',sys.modules['transformers.tokenization_utils_fast'],(directory,changed,permitted))


def test_actual_checkpoint_tokenizer_then_full_native_guard_passes_without_model():
    """Fresh public tokenizer factory, both before/after-import native admission."""
    import os,subprocess
    if torch.__version__!='2.9.1+cu128':pytest.skip('separate exact native Torch/CUDA-build CPU-only environment required')
    root=Path('/mnt/nvme/sliders/qualification/models/music3-bd348f9')
    if not root.is_dir():pytest.skip('exact staged native tokenizer/assets not available')
    code=r'''import json,sys,torch
from pathlib import Path
from music3 import native_assets as assets
assert 'transformers' not in sys.modules
before=torch.get_rng_state().clone()
torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True,warn_only=False)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_cudnn_sdp(False);torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
root=Path('/mnt/nvme/sliders/qualification/models/music3-bd348f9')
options=dict(native_numerical_profile=assets.PROFILE,pg_source_profile=assets.PG_PROFILE,
    penalty_input_domain='continuous',base_precision='bfloat16',device='cuda:0',model_dir=str(root),
    manifest=str(root/'qualification-manifest.json'),
    manifest_sha256='6f08547501d9d38294c91131d22b162fc5d04c683dd258d3b408522558ced7f2')
first=assets.verify_native(options)
assert 'transformers' not in sys.modules
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained(str(root/'tokenizer'),local_files_only=True,trust_remote_code=False)
assert type(tokenizer).__module__+'.'+type(tokenizer).__name__=='transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer'
assert tokenizer.convert_tokens_to_ids('<|audio_start|>')==151669
second=assets.verify_native(options)
assert first==second and second['runtime_source_files']>3000
assert torch.equal(before,torch.get_rng_state()) and not torch.cuda.is_initialized()
print(json.dumps(dict(preimport_source_guard=True,actual_tokenizer_factory=True,postimport_source_guard=True,
    alias_count=sum(name.startswith('transformers.') and vars(module).get('__file__') is None for name,module in sys.modules.items()),
    caller_RNG_exact=True,CUDA_initialized=False)))
'''
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
    env.pop('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE',None);env.pop('NVIDIA_TF32_OVERRIDE',None)
    result=subprocess.run([sys.executable,'-B','-I','-c',code],env=env,capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout+result.stderr
    receipt=json.loads(result.stdout.strip().splitlines()[-1])
    assert receipt==dict(preimport_source_guard=True,actual_tokenizer_factory=True,postimport_source_guard=True,
        alias_count=215,caller_RNG_exact=True,CUDA_initialized=False)


def test_alias_delegation_outside_owned_initializer_refuses(aliases,monkeypatch):
    name='transformers.tokenization_utils_fast';module=sys.modules[name]
    getter=vars(module)['__getattr__']
    monkeypatch.setattr(getter,'__code__',getter.__code__.replace(co_filename='/outside/transformers/__init__.py'))
    with pytest.raises(ValueError,match='delegation'):check(name,module,aliases)


def test_imported_canonical_target_wrong_origin_refuses(aliases,monkeypatch):
    import importlib
    target=importlib.import_module('transformers.tokenization_utils_tokenizers')
    monkeypatch.setattr(target,'__file__','/outside/transformers/tokenization_utils_tokenizers.py')
    with pytest.raises(ValueError,match='canonical alias target origin'):
        check('transformers.tokenization_utils_fast',sys.modules['transformers.tokenization_utils_fast'],aliases)


def test_same_code_and_closure_with_replaced_globals_refuses(aliases,monkeypatch):
    name='transformers.tokenization_utils_fast';module=sys.modules[name]
    original=vars(module)['__getattr__']
    replaced=types.FunctionType(original.__code__,dict(original.__globals__,getattr=lambda *args:'injected delegation'),
        original.__name__,original.__defaults__,original.__closure__)
    assert replaced('PreTrainedTokenizerFast')=='injected delegation'
    monkeypatch.setattr(module,'__getattr__',replaced)
    with pytest.raises(ValueError,match='delegation'):check(name,module,aliases)


@pytest.mark.parametrize('shadow',['getattr','dir','importlib','logger','__builtins__'])
def test_actual_initializer_delegation_globals_refuse_shadowing(aliases,monkeypatch,shadow):
    namespace=aliases[2][2]
    monkeypatch.setitem(namespace,shadow,lambda *args:None)
    with pytest.raises(ValueError,match='global delegation'):
        assets._transformers_aliases(aliases[0],aliases[1])


def test_same_globals_with_previously_captured_wrong_builtins_refuses(aliases,monkeypatch):
    import builtins
    name='transformers.tokenization_utils_fast';module=sys.modules[name]
    original=vars(module)['__getattr__'];namespace=original.__globals__
    saved=namespace['__builtins__']
    try:
        namespace['__builtins__']=dict(vars(builtins),getattr=lambda *args:'captured injected builtin')
        replaced=types.FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
    finally:namespace['__builtins__']=saved
    assert replaced.__globals__ is namespace and replaced.__builtins__ is not vars(builtins)
    assert replaced('PreTrainedTokenizerFast')=='captured injected builtin'
    monkeypatch.setattr(module,'__getattr__',replaced)
    with pytest.raises(ValueError,match='delegation'):check(name,module,aliases)
