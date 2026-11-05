"""CPU source-policy and RNG evidence, never a pretrained/audio qualification."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from music3 import lifecycle as product
from music3 import native_assets as assets


@pytest.fixture
def strict_math(monkeypatch):
    precision=torch.get_float32_matmul_precision()
    deterministic=torch.are_deterministic_algorithms_enabled()
    warn=torch.is_deterministic_algorithms_warn_only_enabled()
    flags=[(torch.backends.cudnn,'benchmark',False),(torch.backends.cudnn,'deterministic',True),
        (torch.backends.cudnn,'allow_tf32',False),(torch.backends.cuda.matmul,'allow_tf32',False),
        (torch.backends.cuda.matmul,'allow_fp16_reduced_precision_reduction',False),
        (torch.backends.cuda.matmul,'allow_bf16_reduced_precision_reduction',False)]
    saved=[(owner,key,getattr(owner,key)) for owner,key,_ in flags]
    backends=[(torch.backends.cuda.enable_flash_sdp,torch.backends.cuda.flash_sdp_enabled,False),
        (torch.backends.cuda.enable_mem_efficient_sdp,torch.backends.cuda.mem_efficient_sdp_enabled,False),
        (torch.backends.cuda.enable_cudnn_sdp,torch.backends.cuda.cudnn_sdp_enabled,False),
        (torch.backends.cuda.enable_math_sdp,torch.backends.cuda.math_sdp_enabled,True),
        (torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp,torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed,False)]
    old_backends=[(setter,getter()) for setter,getter,_ in backends]
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True,warn_only=False)
    for owner,key,value in flags:setattr(owner,key,value)
    for setter,_,value in backends:setter(value)
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    monkeypatch.delenv('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE',raising=False)
    monkeypatch.delenv('NVIDIA_TF32_OVERRIDE',raising=False)
    try:yield dict(native_numerical_profile=assets.PROFILE)
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.use_deterministic_algorithms(deterministic,warn_only=warn)
        for owner,key,value in saved:setattr(owner,key,value)
        for setter,value in old_backends:setter(value)


def forbidden(*args,**kwargs):raise AssertionError('unowned allocation or all-device seeding')


def test_actual_policy_identity_and_eager_attention_are_inspected_without_cuda(strict_math):
    model=SimpleNamespace(config=SimpleNamespace(_attn_implementation='eager'))
    identity=assets.numerical_profile(strict_math,model)
    assert identity==assets.numerical_profile(strict_math,model)
    assert identity['actual_flags']['deterministic_algorithms']
    assert identity['attention']['transformers_implementation']=='eager'
    model.config._attn_implementation='sdpa'
    with pytest.raises(ValueError,match='attention implementation'):
        assets.numerical_profile(strict_math,model)
    with pytest.raises(ValueError,match='explicit'):
        assets.numerical_profile({})
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('setting',['deterministic','warn','benchmark','cudnn','tf32','tf32-cudnn',
    'fp16','bf16','flash','efficient','cudnn-sdp','math','math-low','workspace','forced','nvidia'])
def test_changed_policy_refuses_without_fixing_caller_settings(strict_math,monkeypatch,setting):
    if setting=='deterministic':torch.use_deterministic_algorithms(False)
    elif setting=='warn':torch.use_deterministic_algorithms(True,warn_only=True)
    elif setting=='benchmark':torch.backends.cudnn.benchmark=True
    elif setting=='cudnn':torch.backends.cudnn.deterministic=False
    elif setting=='tf32':torch.backends.cuda.matmul.allow_tf32=True
    elif setting=='tf32-cudnn':torch.backends.cudnn.allow_tf32=True
    elif setting=='fp16':torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=True
    elif setting=='bf16':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
    elif setting=='flash':torch.backends.cuda.enable_flash_sdp(True)
    elif setting=='efficient':torch.backends.cuda.enable_mem_efficient_sdp(True)
    elif setting=='cudnn-sdp':torch.backends.cuda.enable_cudnn_sdp(True)
    elif setting=='math':torch.backends.cuda.enable_math_sdp(False)
    elif setting=='math-low':torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
    elif setting=='workspace':monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':16:8')
    elif setting=='forced':monkeypatch.setenv('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE','1')
    elif setting=='nvidia':monkeypatch.setenv('NVIDIA_TF32_OVERRIDE','1')
    with pytest.raises(ValueError,match='strict numerical policy|forced TF32'):
        assets.numerical_profile(strict_math)
    assert not torch.cuda.is_initialized()


@pytest.fixture
def cuda_generators(monkeypatch):
    generators=tuple(torch.Generator().manual_seed(101+i) for i in range(3))
    visited=[]
    monkeypatch.setattr(torch.cuda,'default_generators',generators)
    monkeypatch.setattr(torch.cuda,'get_rng_state',lambda device:(visited.append(('get',device)),generators[device].get_state())[1])
    monkeypatch.setattr(torch.cuda,'set_rng_state',lambda state,device:(visited.append(('set',device)),generators[device].set_state(state))[1])
    for owner,name in ((torch,'manual_seed'),(torch.cuda,'manual_seed'),(torch.cuda,'manual_seed_all'),
                       (torch.cuda,'seed_all'),(torch.cuda,'device_count')):
        monkeypatch.setattr(owner,name,forbidden)
    return generators,visited


@pytest.mark.parametrize('fail',[False,True])
def test_entire_native_factory_scopes_cpu_and_only_requested_cuda(monkeypatch,cuda_generators,fail):
    generators,visited=cuda_generators
    before=torch.get_rng_state().clone();states=[g.get_state().clone() for g in generators]
    expected=torch.rand(4,generator=torch.Generator().manual_seed(7))
    monkeypatch.setattr(product,'validate',lambda options:None)
    def construct(options):
        assert torch.equal(torch.rand(4),expected)
        assert torch.equal(torch.rand(4,generator=generators[1]),expected)
        assert torch.equal(generators[0].get_state(),states[0])
        assert torch.equal(generators[2].get_state(),states[2])
        if fail:raise RuntimeError('provider critic game failed')
        return 'actual entire factory scope'
    monkeypatch.setattr(product,'_prepare',construct)
    if fail:
        with pytest.raises(RuntimeError,match='provider critic game'):product.prepare({'device':'cuda:1'})
    else:assert product.prepare({'device':'cuda:1'})=='actual entire factory scope'
    assert torch.equal(before,torch.get_rng_state())
    assert all(torch.equal(g.get_state(),s) for g,s in zip(generators,states))
    assert visited==[('get',1),('set',1)] and not torch.cuda.is_initialized()


@pytest.mark.parametrize('device',['cuda','meta','mps'])
def test_ambiguous_device_refuses_before_rng(cuda_generators,device):
    _,visited=cuda_generators
    with pytest.raises(ValueError,match='explicit CUDA device index'):
        with product.native_initialization_rng({'device':device}):pytest.fail('constructed')
    assert visited==[]


def test_native_preflight_refusal_precedes_fork_and_model(monkeypatch,cuda_generators):
    _,visited=cuda_generators
    def refuse(options):raise ValueError('native source or math refusal')
    monkeypatch.setattr(product,'validate',refuse)
    monkeypatch.setattr(product,'_prepare',forbidden)
    with pytest.raises(ValueError,match='native source or math'):product.prepare({'device':'cuda:1'})
    assert visited==[]


@pytest.mark.parametrize('fail',[False,True])
def test_native_inference_factory_restores_only_owned_generators(monkeypatch,cuda_generators,fail):
    generators,visited=cuda_generators
    before=torch.get_rng_state().clone();states=[g.get_state().clone() for g in generators]
    expected=torch.rand(4,generator=torch.Generator().manual_seed(13))
    monkeypatch.setattr(product,'validate_inference',lambda options,spec:None)
    monkeypatch.setattr(product,'_inference_options',lambda options,spec:dict(device='cuda:2',seed=13))
    def construct(options,spec):
        assert torch.equal(torch.rand(4),expected)
        assert torch.equal(torch.rand(4,generator=generators[2]),expected)
        if fail:raise RuntimeError('native inference failed')
        return 'inference without training construction'
    monkeypatch.setattr(product,'_prepare_inference',construct)
    if fail:
        with pytest.raises(RuntimeError,match='native inference failed'):product.prepare_inference({},None)
    else:assert product.prepare_inference({},None)=='inference without training construction'
    assert torch.equal(before,torch.get_rng_state())
    assert all(torch.equal(g.get_state(),s) for g,s in zip(generators,states))
    assert visited==[('get',2),('set',2)]


def test_cpu_native_scope_never_inspects_cuda(cuda_generators):
    _,visited=cuda_generators
    before=torch.get_rng_state().clone()
    with product.native_initialization_rng({'device':'cpu','seed':17}):first=torch.rand(5)
    with product.native_initialization_rng({'device':'cpu','seed':17}):assert torch.equal(first,torch.rand(5))
    assert torch.equal(before,torch.get_rng_state()) and visited==[]


def test_changed_observation_policy_refuses_before_body(strict_math):
    model=torch.nn.Linear(2,2).eval().requires_grad_(False)
    model.config=SimpleNamespace(_attn_implementation='eager')
    with product.observe_native(model,native_options=strict_math):pass
    torch.backends.cuda.enable_flash_sdp(True)
    reached=False
    with pytest.raises(ValueError,match='strict numerical policy'):
        with product.observe_native(model,native_options=strict_math):reached=True
    assert not reached and not torch.cuda.is_initialized()


def test_old_default_package_archive_bytes_and_core_pin_remain_exact():
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'cohorts/pg094-response-ddb-20261104-files.json').read_text())
    old=root/'cohorts/pg094-response-ddb-20261104'
    assert manifest['product_source']=='38733953d911018918671df0ba3ede89ca1c4734'
    assert len(manifest['files'])==20
    for name,digest in manifest['files'].items():assert hashlib.sha256((old/name).read_bytes()).hexdigest()==digest
    assert 'ddb41423bcefc1fe76cd94d0f9b367353b3ea137' in (old/'pyproject.toml').read_text()
    assert assets.native_sources()['lifecycle.py']==hashlib.sha256((root/'music3/lifecycle.py').read_bytes()).hexdigest()


@pytest.mark.parametrize('domain',['auto','continuous'])
def test_fresh_process_actual_fixture_state_and_2_vs_1_plus_1(domain):
    """Actual small Qwen3/provider/critic/prior/streams; native weights stay absent.

    The native whole-factory route wraps the existing narrow CPU fixture at the
    pretrained/tokenizer boundary. Candidate-only geometry does not claim4096
    or acoustic qualification. Different caller seeds/PYTHONHASHSEED must agree.
    """
    profile='release-0.9.5'
    from particle_sliders.pg09 import _released_sources
    try:_released_sources(profile)
    except RuntimeError:pytest.skip('separate exact '+profile+' environment required')
    code=r'''
from copy import deepcopy
import json,os,torch
from music3 import lifecycle as product
from particle_sliders.pg09 import _hash
validate=product.validate
build=product._build_native
prepare=product._prepare
profile=os.environ['MUSIC3_TEST_PROFILE']
domain=os.environ['MUSIC3_TEST_DOMAIN']
# Only the source gate/pretrained-tokenizer boundary is replaced; the actual
# shared CPU adapter/provider/critic/prior/game and update/resume math execute.
product._validate_model=lambda options:None
product.validate=lambda options:None
product._prepare=lambda options:prepare(dict(options,dummy=True,width=32,base_precision='float32',pg_source_profile=profile,penalty_input_domain=domain))
torch.random.default_generator.manual_seed(int(os.environ['CALLER_SEED']))
torch.rand(37);before=torch.get_rng_state().clone()
binding=product.prepare(dict(device='cpu',seed=7,steps=2,batch=2))
assert torch.equal(before,torch.get_rng_state())
try:
 game=binding.game;initial=game.state_dict()
 keys=('network','critic','prior','spread','streams','row_orders','row_cursors','exposures','model_cpu_rng','model_cuda_rng','config')
 hashes={key:_hash(initial[key]) for key in keys}
 first=game.step();one=deepcopy(game.state_dict());second=game.step();final=_hash(game.state_dict())
 game.load_state_dict(one);assert game.step()==second and _hash(game.state_dict())==final
finally:binding.close()
assert torch.equal(before,torch.get_rng_state()) and not torch.cuda.is_initialized()
print(json.dumps(dict(initial=hashes,first=first,second=second,final=final,exact_2_vs_1_plus_1=True,caller_RNG_restored=True,CUDA_initialized=False),sort_keys=True))
'''
    outputs=[]
    for caller,hashseed in [('1234','19'),('98765','403')]:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONHASHSEED=hashseed,CALLER_SEED=caller,
            MUSIC3_TEST_PROFILE=profile,MUSIC3_TEST_DOMAIN=domain,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        result=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True,timeout=90)
        assert result.returncode==0,result.stdout+result.stderr
        outputs.append(json.loads(result.stdout.strip().splitlines()[-1]))
    assert outputs[0]==outputs[1]
