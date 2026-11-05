"""Actual 4096-vector CPU geometry and public packets, with random small host.

This is NOT the genuine36-layer/200000-vocabulary pretrained Music3 model.
Only pretrained loading/tokenization/native-device admission are replaced.
The real routed adapter, provider, critic, PG0.9.5 game, resume and inference run.
"""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from music3 import lifecycle as product
from particle_sliders.inference import open_inference
from particle_sliders.lifecycle import SessionConfig
from particle_sliders.pg09 import _hash, _released_sources
from particle_sliders.response import INFERENCE, open_response

from test_native_profile import strict_math


@pytest.fixture
def candidate_fixture(monkeypatch,strict_math):
    try:_released_sources('release-0.9.5')
    except RuntimeError:pytest.skip('exact published PG0.9.5 environment required')
    build=product._build_native
    monkeypatch.setattr(product,'_validate_model',lambda options:None)
    actual_capabilities=product.capabilities
    def fixture_capabilities(options):
        return replace(actual_capabilities(options),base_precisions=('float32',),
            qualification='random one-layer4096-vector CPU geometry, no pretrained/native audio')
    monkeypatch.setattr(product,'BACKEND',replace(product.BACKEND,configure_capabilities=fixture_capabilities))
    monkeypatch.setattr(product,'INFERENCE',replace(product.INFERENCE,configure_capabilities=fixture_capabilities))
    # The actual source gate inside build, adapter and game still verifies PG0.9.5.
    # Random one-layer Qwen3 is4096-wide but has only256 synthetic token IDs.
    def boundary(options,**kwargs):
        return build(dict(options,dummy=True,device='cpu',base_precision='float32'),**kwargs)
    monkeypatch.setattr(product,'_build_native',boundary)
    from particle_sliders import music3_native
    def fixture_tokens(tokenizer,text,device):
        token=29 if 'crisp' in text else 11
        ids=torch.tensor([[token,13,17]],device=device)
        return ids,torch.ones_like(ids)
    monkeypatch.setattr(music3_native,'tokenize_text',fixture_tokens)
    monkeypatch.setattr(music3_native,'assert_audio_start',lambda *args,**kwargs:None)
    return SessionConfig('music3','music3-4096-random-CPU-geometry-only',device='cpu',
        base_precision='float32',width=4096,steps=2,batch=2,seed=7,ema_decay=.9,
        pg_source_profile='release-0.9.5',penalty_input_domain='continuous',
        native_numerical_profile=strict_math['native_numerical_profile'],
        rows=[dict(neutral='soft piano',positive='crisp piano',lyrics=''),
              dict(neutral='quiet piano',positive='crisp quiet piano',lyrics='')])


def test_actual_4096_geometry_exact_fresh_common_resume(candidate_fixture,tmp_path):
    cfg=candidate_fixture
    before=torch.get_rng_state().clone()
    with open_response(cfg) as full:
        assert full.game.backend.model.config.hidden_size==4096
        assert full.game.backend.model.config.vocab_size==256
        full.train();expected=full.checkpoint()
    with open_response(cfg) as first:
        first.train(stop_after=1);first.checkpoint(tmp_path/'step1.pt')
    with open_response(cfg,resume=tmp_path/'step1.pt') as resumed:
        resumed.train();assert _hash(resumed.checkpoint())==_hash(expected)
    assert torch.equal(before,torch.get_rng_state()) and not torch.cuda.is_initialized()


@pytest.mark.parametrize('selection',['raw','ema'])
def test_actual_4096_public_v1_v2_numeric_responses_without_training_constructors(candidate_fixture,monkeypatch,selection,tmp_path):
    from particle_sliders import pg09,pg09_audio
    with open_response(candidate_fixture) as run:
        run.train();packet=run.export_inference(selection=selection)
        legacy=run.export(tmp_path/'legacy-v1.pt',selection=selection)
        with run._observation(23):
            if selection=='ema':run.game.network.load_state_dict(run.ema.values)
            expected={strength:run.game.backend._response((11,13,17),scale=strength).detach().clone() for strength in (0.,.5,1.)}
    with open_response(candidate_fixture,artifact=tmp_path/'legacy-v1.pt',operation='response') as v1:
        assert v1.inference_only
        for strength in (0.,.5,1.):
            assert torch.equal(v1.evaluate(0,strength=strength,seed=23),expected[strength])
    def forbidden(*args,**kwargs):raise AssertionError('inference constructed training state')
    monkeypatch.setattr(product,'AudioProvider',forbidden)
    monkeypatch.setattr(pg09.ConditionalPG09,'__init__',forbidden)
    monkeypatch.setattr(pg09_audio,'AudioResponseCritic',forbidden)
    with open_inference(packet,options={'device':'cpu'},operation='response') as native:
        for strength in (0.,.5,1.):
            assert torch.equal(native.response(ids=[11,13,17],strength=strength,seed=23),expected[strength])
        assert expected[0.].shape==(1,4096)
        assert not torch.equal(expected[0.],expected[1.])
        assert not any(hasattr(native,name) for name in ('game','critic','optimizer'))
    assert legacy['schema']==INFERENCE
    assert not torch.cuda.is_initialized()
