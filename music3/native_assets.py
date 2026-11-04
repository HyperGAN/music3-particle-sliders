"""Exact Music3 BF16 Qwen3 candidate, caller-owned numerical policy and assets.

Metadata checks and complete local byte hashing precede any model allocation.
No download, profile fallback, projection, implicit flag mutation or GPU query.
"""
from importlib import metadata
import hashlib
import json
import os
from pathlib import Path

PROFILE='music3-bf16-eager-strict-v1'
PG_PROFILE='source-801ae1e-continuous-v1'
MODEL_REVISION='bd348f9c49ea3c1b39f33ace3436f8fad435f24e'
DIFFUSERS_COMMIT='d5baa4fb548294f47dbca49890abd4b291204c60'
LOCK=Path(__file__).with_name('native-assets.lock.json')
LOCK_SHA256='02d927d1fb7611e74273da8006f8fe5ccb288508958b84e9e94ba7a3564152ff'
VERSIONS={'torch':'2.9.1','transformers':'5.15.0','diffusers':'0.40.0.dev0',
          'huggingface-hub':'1.27.0','safetensors':'0.8.0','numpy':'2.2.6','tokenizers':'0.22.2'}
ARCHITECTURE=dict(hidden_size=4096,num_hidden_layers=36,num_attention_heads=32,
    num_key_value_heads=8,head_dim=128,intermediate_size=12288,vocab_size=200000,max_position_embeddings=10240)


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def source_files(root):
    root=Path(root)
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}


def native_sources():
    root=Path(__file__).parent
    return {name:sha(root/name) for name in ('lifecycle.py','response_source.py','native_assets.py',
        'native-assets.lock.json','response-core.lock.json','learning_rates.py')}


def numerical_profile(options,model=None):
    """Inspect strict caller-owned flags, never silently repair them."""
    import torch
    if options.get('native_numerical_profile')!=PROFILE:
        raise ValueError('native Music3 requires explicit '+PROFILE)
    actual=dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        flash_sdp=torch.backends.cuda.flash_sdp_enabled(),memory_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
        cudnn_sdp=torch.backends.cuda.cudnn_sdp_enabled(),math_sdp=torch.backends.cuda.math_sdp_enabled(),
        math_sdp_low_precision_reduction=torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    expected=dict(deterministic_algorithms=True,deterministic_warn_only=False,
        cudnn_benchmark=False,cudnn_deterministic=True,tf32_matmul=False,tf32_cudnn=False,
        float32_matmul_precision='highest',fp16_reduced_precision_reduction=False,
        bf16_reduced_precision_reduction=False,flash_sdp=False,memory_efficient_sdp=False,
        cudnn_sdp=False,math_sdp=True,math_sdp_low_precision_reduction=False,cublas_workspace_config=':4096:8')
    if actual!=expected:
        raise ValueError('native Music3 strict numerical policy differs: '+', '.join(k for k in expected if actual[k]!=expected[k]))
    overrides={k:os.environ.get(k) for k in ('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE','NVIDIA_TF32_OVERRIDE')}
    if any(v not in (None,'0') for v in overrides.values()):
        raise ValueError('native Music3 refuses forced TF32 environment overrides')
    if model is not None and getattr(model.config,'_attn_implementation',None)!='eager':
        raise ValueError('native Music3 requires its exact Qwen3 eager attention implementation')
    return dict(profile=PROFILE,actual_flags=actual,environment_overrides=overrides,
        attention=dict(transformers_implementation='eager',
            source='transformers.models.qwen3.modeling_qwen3.eager_attention_forward'))


def verify_native(options):
    if (options.get('pg_source_profile'),options.get('penalty_input_domain'))!=(PG_PROFILE,'continuous'):
        raise ValueError('native Music3 requires the explicit current continuous source/domain pair')
    if options.get('base_precision')!='bfloat16' or not options.get('device','').startswith('cuda:'):
        raise ValueError('native Music3 requires explicit CUDA device and bfloat16 base precision')
    import torch
    if torch.__version__!='2.9.1+cu128' or torch.version.cuda!='12.8':
        raise ValueError('native Music3 requires actual Torch2.9.1+cu128/CUDA12.8')
    policy=numerical_profile(options)
    versions={name:metadata.version(name) for name in VERSIONS}
    if versions!=VERSIONS:raise ValueError('native Music3 runtime versions differ from candidate')
    direct=json.loads(metadata.distribution('diffusers').read_text('direct_url.json') or '{}')
    if (direct.get('url'),direct.get('vcs_info',{}).get('vcs'),direct.get('vcs_info',{}).get('commit_id'))!=('https://github.com/huggingface/diffusers.git','git',DIFFUSERS_COMMIT) or 'dir_info' in direct or 'archive_info' in direct:
        raise ValueError('native Music3 requires exact immutable Diffusers source for native caption/lyrics packing')
    if sha(LOCK)!=LOCK_SHA256:raise ValueError('native Music3 source asset lock bytes differ')
    lock=json.loads(LOCK.read_text())
    if (lock.get('schema'),lock.get('revision'),lock.get('architecture'),lock.get('tensor_count'),lock.get('elements'))!=('music3-native-assets-v1',MODEL_REVISION,ARCHITECTURE,399,8584475648):
        raise ValueError('native Music3 asset source descriptor differs')
    manifest=Path(options.get('manifest',''))
    if options.get('manifest_sha256')!=lock['staged_manifest_sha256'] or not manifest.is_file() or sha(manifest)!=lock['staged_manifest_sha256']:
        raise ValueError('native Music3 requires its exact approved staged manifest bytes')
    root=Path(options.get('model_dir',''))
    actual={str(p.relative_to(root)):p for folder in ('language_model','tokenizer')
        for p in (root/folder).rglob('*') if p.is_file()}
    if set(actual)!=set(lock['files']):raise ValueError('native Music3 LM/tokenizer file inventory differs')
    for name,expected in lock['files'].items():
        path=actual[name]
        if path.stat().st_size!=expected['bytes'] or sha(path)!=expected['sha256']:
            raise ValueError('native Music3 complete cached file bytes differ: '+name)
    config=json.loads((root/'language_model/config.json').read_text())
    if {k:config.get(k) for k in ARCHITECTURE}!=ARCHITECTURE or config.get('dtype',config.get('torch_dtype'))!='bfloat16':
        raise ValueError('native Music3 genuine Qwen3 architecture/precision differs')
    # Import ownership and source byte closure supplement distribution versions.
    import importlib.util,sys
    owned={}
    for package,distname in (('transformers','transformers'),('diffusers','diffusers')):
        directory=Path(metadata.distribution(distname).locate_file(package)).resolve()
        spec=importlib.util.find_spec(package)
        if spec is None or spec.origin is None or Path(spec.origin).resolve()!=directory/'__init__.py':
            raise ValueError('native Music3 package import origin differs: '+package)
        for name,module in tuple(sys.modules.items()):
            if name.split('.')[0]==package:
                origin=getattr(module,'__file__',None)
                if origin is None or not Path(origin).resolve().is_relative_to(directory):
                    raise ValueError('native Music3 already-imported package origin differs: '+name)
        owned[package]=dict(package=str(directory))
    for name,digest in lock['runtime_sources_sha256'].items():
        distribution_name,relative=name.split(':',1)
        path=Path(metadata.distribution(distribution_name).locate_file(relative))
        if not path.is_file() or sha(path)!=digest:
            raise ValueError('native Music3 exact runtime source bytes differ: '+name)
    return dict(schema='music3-native-assets-v1',model_revision=MODEL_REVISION,
        asset_lock_sha256=sha(LOCK),manifest_sha256=lock['staged_manifest_sha256'],
        architecture=ARCHITECTURE,runtime_versions=versions,actual_torch=torch.__version__,cuda_build=torch.version.cuda,
        numerical_profile=policy,source_sha256=native_sources(),runtime_source_files=len(lock['runtime_sources_sha256']),runtime_sources_closure_sha256=hashlib.sha256(json.dumps(lock['runtime_sources_sha256'],sort_keys=True).encode()).hexdigest(),
        qualification='exact native source candidate; GPU/resume/media quality qualification is separate')


def verify_loaded_model(model,options):
    """Every genuine BF16 checkpoint key/shape/value before adapter attachment."""
    import torch
    from safetensors import safe_open
    if {k:getattr(model.config,k,None) for k in ARCHITECTURE}!=ARCHITECTURE:
        raise ValueError('loaded Music3 genuine Qwen3 architecture differs')
    numerical_profile(options,model)
    from transformers import Qwen3ForCausalLM
    if type(model) is not Qwen3ForCausalLM :
        raise ValueError('native Music3 requires its exact pristine Qwen3 model before attachment/audit')
    root=Path(options['model_dir'])/'language_model'
    index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    values=model.state_dict();digest=hashlib.sha256()
    if set(values)!=set(index) or len(values)!=399:
        raise ValueError('loaded Music3 complete checkpoint tensor inventory differs')
    elements=0
    for shard in sorted(set(index.values())):
        with safe_open(root/shard,framework='pt',device='cpu') as archive:
            expected={key for key,name in index.items() if name==shard}
            if set(archive.keys())!=expected:raise ValueError('Music3 shard/index keys differ')
            for key in sorted(expected):
                value=values[key];checkpoint=archive.get_tensor(key)
                if str(value.device)!=options['device'] or value.dtype!=torch.bfloat16 or checkpoint.dtype!=torch.bfloat16:
                    raise ValueError('loaded Music3 device/BF16 checkpoint precision differs: '+key)
                actual=value.detach().cpu().contiguous()
                if actual.shape!=checkpoint.shape or not torch.equal(actual,checkpoint):
                    raise ValueError('loaded Music3 checkpoint value differs: '+key)
                elements+=actual.numel();digest.update(key.encode());digest.update(actual.reshape(-1).view(torch.uint8).numpy().tobytes())
    if elements!=8584475648:raise ValueError('loaded Music3 checkpoint element count differs')
    buffers={}
    for name,value in model.named_buffers():
        value=value.detach().cpu().contiguous();content=hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
        buffers[name]=dict(shape=list(value.shape),dtype=str(value.dtype),sha256=content)
    return dict(complete_checkpoint_tensors=399,elements=elements,all_values_exact=True,
        frozen_base_sha256=digest.hexdigest(),native_buffers=buffers)
