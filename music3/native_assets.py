"""Exact Music3 BF16 Qwen3 candidate, caller-owned numerical policy and assets.

Metadata checks and complete local byte hashing precede any model allocation.
No download, profile fallback, projection, implicit flag mutation or GPU query.
"""
from importlib import metadata
from importlib import import_module as _NATIVE_IMPORT_MODULE
from builtins import getattr as _NATIVE_GETATTR, dir as _NATIVE_DIR
import hashlib
import json
import os
from pathlib import Path

PROFILE='music3-bf16-eager-strict-pg095-v1'
PG_PROFILE='release-0.9.5'
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
        'native-assets.lock.json','response-core.lock.json','learning_rates.py','native.py','response_adapter.py','native-provenance.json')}


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


def _transformers_aliases(directory, runtime_sources):
    """Only virtual modules emitted by this exact locked Transformers initializer.

    The three literal aliases and per-model image aliases are source-derived;
    missing origins are never accepted for arbitrary imported modules.
    """
    import builtins, importlib, logging, sys, types
    directory = Path(directory).resolve()
    initializer = directory / '__init__.py'
    expected = runtime_sources.get('transformers:transformers/__init__.py')
    if expected is None or sha(initializer) != expected:
        raise ValueError('native Music3 exact alias initializer source differs')
    aliases = {
        'transformers.tokenization_utils_fast': '.tokenization_utils_tokenizers',
        'transformers.tokenization_utils': '.tokenization_utils_sentencepiece',
        'transformers.image_processing_utils_fast': '.image_processing_backends',
    }
    for key in runtime_sources:
        owner, relative = key.split(':', 1)
        path = Path(relative)
        if (owner == 'transformers' and len(path.parts) == 4
                and path.parts[:2] == ('transformers', 'models')
                and path.name.startswith('image_processing_') and path.suffix == '.py'):
            target = '.models.' + path.parent.name + '.' + path.stem
            aliases['transformers' + target + '_fast'] = target
    codes = []
    def collect(code):
        codes.append(code)
        for value in code.co_consts:
            if isinstance(value, types.CodeType):
                collect(value)
    collect(compile(initializer.read_bytes(), str(initializer), 'exec', dont_inherit=True))
    pick = lambda name: [code for code in codes if code.co_qualname == name]
    lambdas = sorted(pick('_create_module_alias.<locals>.<lambda>'), key=lambda code: code.co_firstlineno)
    if len(lambdas) != 2:
        raise ValueError('native Music3 locked alias initializer structure differs')
    expected_codes = dict(getattr=lambdas[0], dir=lambdas[1],
        target=pick('_create_module_alias.<locals>._get_target')[0],
        image_getattr=pick('getattr_factory.<locals>._getattr')[0],
        alias_factory=pick('_create_module_alias')[0], image_factory=pick('getattr_factory')[0])
    # Transformers replaces its public module with _LazyModule. The initializer
    # functions retain the ORIGINAL namespace; anchor and inspect that actual
    # namespace rather than assuming it is vars(current public module).
    anchor = vars(sys.modules['transformers.tokenization_utils_fast']).get('__dir__')
    if type(anchor) is not types.FunctionType:
        raise ValueError('native Music3 alias initializer namespace anchor differs')
    namespace = anchor.__globals__
    if (namespace.get('__name__') != 'transformers'
            or Path(namespace.get('__file__', '')).resolve() != initializer
            or namespace.get('__builtins__') is not vars(builtins)
            or builtins.getattr is not _NATIVE_GETATTR or builtins.dir is not _NATIVE_DIR
            or any(name in namespace for name in ('getattr', 'dir'))
            or namespace.get('importlib') is not importlib
            or importlib.import_module is not _NATIVE_IMPORT_MODULE
            or namespace.get('logger') is not logging.getLogger('transformers')):
        raise ValueError('native Music3 alias initializer global delegation differs')
    for name, key in (('_create_module_alias', 'alias_factory'), ('getattr_factory', 'image_factory')):
        value = namespace.get(name)
        if (type(value) is not types.FunctionType or value.__globals__ is not namespace
                or value.__builtins__ is not vars(builtins)
                or value.__code__ != expected_codes[key]
                or Path(value.__code__.co_filename).resolve() != initializer):
            raise ValueError('native Music3 alias initializer factory globals differ')
    return aliases, expected_codes, namespace


def _verify_transformers_alias(name, module, directory, runtime_sources, permitted):
    """Inspect code, closure, target bytes and actual module without invoking it."""
    import builtins
    import importlib
    import sys
    import types
    aliases, codes, namespace = permitted
    target = aliases.get(name)
    if target is None:
        raise ValueError('native Music3 unknown originless imported module: ' + name)
    if (type(module) is not types.ModuleType or sys.modules.get(name) is not module
            or vars(sys.modules['transformers']).get(name.rsplit('.', 1)[-1]) is not module):
        raise ValueError('native Music3 exact registered alias module differs: ' + name)
    values = vars(module)
    expected_keys = {'__name__', '__doc__', '__package__', '__loader__', '__spec__',
                     '__file__', '__getattr__', '__dir__'}
    if (type(module) is not types.ModuleType or set(values) != expected_keys
            or values['__name__'] != name or values['__file__'] is not None
            or any(values[key] is not None for key in ('__package__', '__loader__', '__spec__'))
            or values['__doc__'] != f'Alias module for backward compatibility with `{target}`.'):
        raise ValueError('native Music3 exact virtual alias module identity differs: ' + name)
    initializer = str(Path(directory).resolve() / '__init__.py')
    def function(value, code, closed):
        return (type(value) is types.FunctionType and value.__code__ == code
            and str(Path(value.__code__.co_filename).resolve()) == initializer
            and value.__defaults__ is None and value.__kwdefaults__ is None
            and value.__globals__ is namespace
            and value.__builtins__ is vars(builtins)
            and len(value.__closure__ or ()) == len(closed)
            and all(type(cell.cell_contents) is type(expected) and cell.cell_contents == expected
                    for cell, expected in zip(value.__closure__ or (), closed)))
    getter, directory_fn = values['__getattr__'], values['__dir__']
    cells = tuple(cell.cell_contents for cell in directory_fn.__closure__ or ()) if type(directory_fn) is types.FunctionType else ()
    get_target = cells[0] if len(cells) == 1 else None
    if (not function(get_target, codes['target'], (target,))
            or not function(directory_fn, codes['dir'], (get_target,))
            or not function(getter, codes['image_getattr'] if name.endswith('_fast') and target.startswith('.models.') else codes['getattr'],
                            (target,) if target.startswith('.models.') else (get_target,))):
        raise ValueError('native Music3 exact virtual alias delegation differs: ' + name)
    relative = 'transformers/' + target.lstrip('.').replace('.', '/') + '.py'
    digest = runtime_sources.get('transformers:' + relative)
    source = Path(directory).parent / relative
    if digest is None or not source.is_file() or sha(source) != digest:
        raise ValueError('native Music3 exact virtual alias target source differs: ' + name)
    canonical = sys.modules.get('transformers' + target)
    if canonical is not None:
        origin = vars(canonical).get('__file__')
        spec = vars(canonical).get('__spec__')
        if (not isinstance(origin, str) or Path(origin).resolve() != source.resolve()
                or spec is None or spec.origin is None or Path(spec.origin).resolve() != source.resolve()):
            raise ValueError('native Music3 imported canonical alias target origin differs: ' + name)
    return target


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
        permitted = (_transformers_aliases(directory, lock['runtime_sources_sha256'])
            if package == 'transformers' and package in sys.modules else None)
        for name,module in tuple(sys.modules.items()):
            if name.split('.')[0]==package:
                origin=vars(module).get('__file__')
                if origin is None and package == 'transformers':
                    if permitted is None:
                        raise ValueError('native Music3 originless submodule without actual package initializer: ' + name)
                    _verify_transformers_alias(name, module, directory, lock['runtime_sources_sha256'], permitted)
                    continue
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
