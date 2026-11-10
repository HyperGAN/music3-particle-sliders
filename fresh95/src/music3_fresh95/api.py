"""Explicit fresh/FULL/RAW owner for original Music3 final AR responses, not PCM."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import hashlib,torch
from particle_sliders.pg095 import ConditionalPG095
from . import PROFILE,ownership,policy
from .cache import load_cache,to_device,teacher_identity
from .util import canonical,equal,new_directory,read_packet,save_packet,write_json
FULL_SCHEMA='Music3-explicit-cached-fresh95-FULL-v1'
RAW_SCHEMA='Music3-explicit-cached-fresh95-RAW-v1'
CONSTRUCTOR='new-owner-original-ParticleNetwork-seed-and-critic-seed-plus1-v1'

def original_options(config):
    keys=('steps','batch','seed','width','device','base_precision','checkpointing','max_sequence_length','pg_source_profile','penalty_input_domain','preset','ema_decay','lr','d_lr_mult','prior_lr_mult')
    value={k:config[k]for k in keys};value.update(dummy=config['lane']=='cpu-fixture',host='lm')
    if config['lane']=='native4096':value.update({k:config[k]for k in('model_dir','manifest','manifest_sha256','native_numerical_profile')})
    return value

def configuration(value):
    if not isinstance(value,dict):raise ValueError('explicit Music3 configuration required')
    common={'lane','model_identity','steps','batch','seed','width','device','base_precision','checkpointing','max_sequence_length','pg_source_profile','penalty_input_domain','preset','ema_decay','lr','d_lr_mult','prior_lr_mult'}
    native=value.get('lane')=='native4096'
    if value.get('lane')not in('cpu-fixture','native4096')or set(value)!=(common|({'model_dir','manifest','manifest_sha256','native_numerical_profile'}if native else set())):raise ValueError('exact explicit CPU fixture or native4096 lane/options required')
    if value['ema_decay']is not None:raise ValueError('model-weight EMA refused; RAW only')
    if not isinstance(value['model_identity'],dict)or not value['model_identity']:raise ValueError('exact original whole native/base identity required')
    if(not native and(value['width']!=32 or value['device']!='cpu'or value['base_precision']!='float32')or native and value['width']!=4096):raise ValueError('original32-wide fixture or complete4096 native response required')
    from music3.lifecycle import _validate_model,capabilities,_integer
    from music3.learning_rates import recipe_overrides
    options=original_options(value)
    _validate_model(options)  # Exact original domain, assets, strict numerical/runtime gate; no fallback.
    _integer(options,'steps',4,1,1000000);_integer(options,'batch',2,2,512)
    if type(value['checkpointing'])is not bool or value['preset']not in('bispf','bispf_optimized'):raise ValueError('original checkpoint/preset contract required')
    if any(value[k]is None for k in('lr','d_lr_mult','prior_lr_mult')):raise ValueError('explicit caller rates; no preset migration')
    recipe_overrides(**{k:value[k]for k in('lr','d_lr_mult','prior_lr_mult')});capabilities(options).check('train',precision=value['base_precision'])
    canonical(value);return deepcopy(value)

def intent(mode,resume,cohort,ema_decay):
    if mode not in('fresh','resume')or not isinstance(cohort,str)or not cohort.strip():raise ValueError('explicit fresh/resume/cohort required')
    if mode=='fresh'and resume is not None or mode=='resume'and resume is None:raise ValueError('fresh refuses state; strict resume requires exact new FULL')
    if ema_decay is not None:raise ValueError('weight EMA refused')

def model_identity(model,config):
    from transformers import Qwen3ForCausalLM
    from music3 import lifecycle as original
    if type(model)is not Qwen3ForCausalLM:raise ValueError('exact original Qwen3ForCausalLM bare component required')
    if any('forward'in vars(m)or m._forward_hooks or m._forward_pre_hooks for m in model.modules()):raise ValueError('original pristine unwrapped native host required')
    if config['lane']=='cpu-fixture':
        from particle_sliders.pg09 import _hash
        architecture=dict(hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=16,vocab_size=256,max_position_embeddings=config['max_sequence_length'])
        if any(getattr(model.config,k,None)!=v for k,v in architecture.items())or getattr(model.config,'_attn_implementation',None)!='eager':raise ValueError('exact original random Qwen3 fixture geometry required')
        if any(p.device.type!='cpu'or p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('original fixture CPU FP32 required')
        return dict(fixture='random Qwen3; synthetic token IDs',seed=config['seed'],width=32,frozen_base_sha256=_hash(model.state_dict()))
    from music3.native_assets import verify_loaded_model,verify_native,numerical_profile
    options=original_options(config);files=original._native_record(options)
    loaded=verify_loaded_model(model,options)
    return dict(manifest_sha256=options['manifest_sha256'],revision=original.MODEL_REVISION,files=files,loaded_checkpoint=loaded,
        native_assets=verify_native(options),native_numerical_profile=numerical_profile(options,model),initialization_rng_profile='cpu-and-requested-cuda-full-prepare-v1',frozen_base_sha256=loaded['frozen_base_sha256'])

class ReceiptBackend:
    def __init__(self,adapter,receipt,sources,cache):self.backend=adapter;self.network=adapter.network;self.receipt=deepcopy(receipt);self.sources=deepcopy(sources);self.cache=deepcopy(cache)
    def __getattr__(self,name):return getattr(self.backend,name)
    def metadata(self):return dict(self.backend.metadata(),fresh_initialization=dict(profile=PROFILE,receipt=deepcopy(self.receipt),sources=deepcopy(self.sources)),completed_native_cache=deepcopy(self.cache),teacher_collector_called=False)

class NativeOwner(ownership.Owner):
    def _components(self,request,factory):
        self.model=factory(deepcopy(request))
        if model_identity(self.model,self.config)!=self.config['model_identity']:raise ValueError('actual original native/base identity differs')
        self.attachment=ownership.Attachment(self.model);self.adapter=self.attachment.construct(self.config);self.slider=self.attachment.slider;self.network=self.adapter.network
        if self.network.particles is not self.slider.particles:raise RuntimeError('original bank Parameter identity lost')
    @contextmanager
    def observation(self,seed):
        from particle_sliders.lifecycle import _network_observation
        from music3.lifecycle import observe_native
        self.require_open();device=torch.device(self.config['device']);devices=[device.index]if device.type=='cuda'else[]
        with _network_observation(self.network,seed,devices=devices),observe_native(self.model,self.network,None if self.config['lane']=='cpu-fixture'else original_options(self.config)):yield
    def response(self,*,ids,strength=1.,seed=7):
        from music3.lifecycle import capabilities,_probe_ids
        options=original_options(self.config);capabilities(options).check('response',strength=strength)
        request=_probe_ids(options,dict(ids=ids))
        with self.observation(seed),torch.no_grad():return self.adapter._response(request,scale=strength).detach().clone()

class Training(NativeOwner):
    def __init__(self,config,cache,directory,*,mode,cohort_id,component_factory,owner_admission,resume=None,ema_decay=None):
        super().__init__();intent(mode,resume,cohort_id,ema_decay)
        if not callable(component_factory)or not callable(owner_admission):raise ValueError('genuine component factory/admitting owner required')
        self.directory=Path(directory).absolute()
        if self.directory.exists():raise FileExistsError('new immutable directory required')
        self.runtime_identity=ownership.guard();self.config=configuration(config);self.options=dict(config=self.config,constructor=CONSTRUCTOR)
        self.sources=ownership.sources();self.cohort_id=cohort_id;self.construction=policy.Construction(mode,cohort_id,hashlib.sha256(canonical(self.options).encode()).hexdigest())
        packet=read_packet(resume)if mode=='resume'else None
        if packet is not None:
            keys={'schema','profile','cohort_id','options','sources','initialization','cache_identity','game','ema_decay'}
            if(not isinstance(packet,dict)or set(packet)!=keys or packet['schema']!=FULL_SCHEMA or packet['profile']!=PROFILE or packet['cohort_id']!=cohort_id
                or packet['options']!=self.options or packet['sources']!=self.sources or packet['ema_decay']is not None):raise ValueError('exact whole new profile FULL required; no old cohort reset/migration')
            policy.validate_receipt(packet['initialization'],self.construction,self.options,self.sources)
        prepared,conditions,self.cache_identity=load_cache(cache,config=self.config)
        if packet is not None and packet['cache_identity']!=self.cache_identity:raise ValueError('same completed original cache required')
        request=dict(operation=mode,profile=PROFILE,cohort_id=cohort_id,options=deepcopy(self.options),completed_cache=deepcopy(self.cache_identity),new_teacher_or_tokenizer_calls=0,fresh_PG_initialization_calls=int(mode=='fresh'),ema_decay=None)
        if owner_admission(deepcopy(request))is not True:raise ValueError('owner did not authenticate/admit exact operation')
        try:
            self._components(request,component_factory)
            from music3.lifecycle import native_initialization_rng
            from particle_sliders.pg09_audio import AudioResponseCritic
            options=original_options(self.config);options['seed']+=1
            with native_initialization_rng(options):self.critic=AudioResponseCritic(self.config['width'],conditions.shape[1]).to(self.config['device'])
            self.initialization=policy.initialize(self.slider,self.critic,self.construction,self.options,self.sources)if mode=='fresh'else deepcopy(packet['initialization'])
            new_directory(self.directory);write_json(self.directory/'initialization.json',self.initialization)
            self.backend=ReceiptBackend(self.adapter,self.initialization,self.sources,self.cache_identity)
            self.game=ConditionalPG095(self.backend,to_device(prepared,self.config['device']),self.critic,conditions=conditions.to(self.config['device']),particles=self.slider.particles,
                response_domain='continuous_vector',response_scale=1.0,preset=self.config['preset'],total_steps=self.config['steps'],batch_size=self.config['batch'],seed=self.config['seed'],
                pg_source_profile=self.config['pg_source_profile'],penalty_input_domain=self.config['penalty_input_domain'],recipe_overrides={k:self.config[k]for k in('lr','d_lr_mult','prior_lr_mult')})
            if packet is not None:
                self.game.load_state_dict(packet['game'])
                if not equal(self.game.state_dict(),packet['game']):raise RuntimeError('strict whole game/optimizers/cursors/RNG state differs')
            self.inherited_steps=self.game.completed_steps
            self.entry_FULL=self.save(self.directory/f'FULL-{self.inherited_steps:06d}.pt');self.entry_RAW=self.export(self.directory/f'RAW-{self.inherited_steps:06d}.pt')
        except BaseException as primary:
            try:self.close()
            except BaseException as cleanup:raise BaseExceptionGroup('Music3 entry and cleanup failed',[primary,cleanup])
            raise
    def update(self):
        self.require_open()
        if self.config['lane']=='native4096':
            from music3.native_assets import numerical_profile
            if numerical_profile(original_options(self.config),self.model)!=self.config['model_identity']['native_numerical_profile']:raise ValueError('native numerical policy changed')
        return self.game.step()
    def save(self,path):
        self.require_open();return save_packet(path,dict(schema=FULL_SCHEMA,profile=PROFILE,cohort_id=self.cohort_id,options=deepcopy(self.options),sources=deepcopy(self.sources),initialization=deepcopy(self.initialization),cache_identity=deepcopy(self.cache_identity),game=self.game.state_dict(),ema_decay=None))
    def export(self,path):
        self.require_open();state=self.game.export_state_dict();return save_packet(path,dict(schema=RAW_SCHEMA,profile=PROFILE,cohort_id=self.cohort_id,options=deepcopy(self.options),sources=deepcopy(self.sources),initialization=deepcopy(self.initialization),cache_identity=deepcopy(self.cache_identity),network=state['network'],completed_steps=self.game.completed_steps,selection='raw',ema_decay=None))

class Inference(NativeOwner):
    def __init__(self,raw,*,component_factory,owner_admission):
        super().__init__();packet=read_packet(raw)
        keys={'schema','profile','cohort_id','options','sources','initialization','cache_identity','network','completed_steps','selection','ema_decay'}
        if(not isinstance(packet,dict)or set(packet)!=keys or packet['schema']!=RAW_SCHEMA or packet['profile']!=PROFILE or packet['selection']!='raw'or packet['ema_decay']is not None
            or type(packet['completed_steps'])is not int or packet['completed_steps']<0 or not callable(component_factory)or not callable(owner_admission)):raise ValueError('exact typed new RAW with actual owner required')
        self.runtime_identity=ownership.guard();self.config=configuration(packet['options']['config']);self.options=dict(config=self.config,constructor=CONSTRUCTOR);self.sources=ownership.sources()
        if(packet['options']!=self.options or packet['sources']!=self.sources or packet['cache_identity']['teacher_identity']!=teacher_identity(self.config)or packet['completed_steps']>self.config['steps']):raise ValueError('exact source/lane/config/cache/update identity required')
        construction=policy.Construction('inference',packet['cohort_id'],hashlib.sha256(canonical(self.options).encode()).hexdigest());policy.validate_receipt(packet['initialization'],construction,self.options,self.sources)
        self.initialization=deepcopy(packet['initialization'])
        request=dict(operation='inference',profile=PROFILE,cohort_id=packet['cohort_id'],options=deepcopy(self.options),completed_cache_metadata_only=deepcopy(packet['cache_identity']),new_teacher_or_tokenizer_calls=0,fresh_PG_initialization_calls=0,ema_decay=None)
        if owner_admission(deepcopy(request))is not True:raise ValueError('owner did not authenticate/admit exact native RAW operation')
        try:
            self._components(request,component_factory);self.network.load_state_dict(packet['network'],strict=True)
            if not equal(self.network.state_dict(),packet['network']):raise RuntimeError('strict RAW addednetwork state differs')
            self.completed_steps=packet['completed_steps']
        except BaseException as primary:
            try:self.close()
            except BaseException as cleanup:raise BaseExceptionGroup('Music3 RAW and cleanup failed',[primary,cleanup])
            raise

def open_training(*,config,cache,directory,mode,cohort_id,component_factory,owner_admission,resume=None,ema_decay=None):return Training(config,cache,directory,mode=mode,cohort_id=cohort_id,component_factory=component_factory,owner_admission=owner_admission,resume=resume,ema_decay=ema_decay)
def open_inference(raw,*,component_factory,owner_admission):return Inference(raw,component_factory=component_factory,owner_admission=owner_admission)
