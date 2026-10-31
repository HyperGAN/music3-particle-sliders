"""Music3 native host plumbing; graph, GAN and optimizer execution are shared."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

from .source import verify_core
verify_core()

import torch
from torch import nn
from particle_sliders.fibo_hndl import HNDLParticleProjection, verified_builder
from particle_sliders.lifecycle import native_cleanup
from particle_sliders.native_lora import lora_target_plan
from particle_sliders.pg09 import ConditionalPG09, PG_SOURCE_PROFILE, _hash
from particle_sliders.pg09_audio import Music3ARResponseAdapter, AudioRow, audio_conditions, require_audio_width
verify_core()  # Check origins again after importing the shared implementation.


class Music3HNDLNetwork(nn.Module):
    """Attach the existing shared HNDL projection to genuine Music3 q/k/v/o."""
    def __init__(self, model):
        super().__init__()
        builder, identity = verified_builder()
        self.identity = identity
        self.particles = nn.Parameter(torch.randn(128, 4))
        self.projections = nn.ModuleDict()
        self.scale = 0.
        self._forwards = []
        self._restore_native = native_cleanup(model)
        try:
            model.eval().requires_grad_(False)
            targets = lora_target_plan(model, ['Qwen3Attention'], prefix='lora_te', delimiter='-', train_method='full')
            if not targets:
                raise ValueError('Music3 HNDL found no Qwen3 attention linears')
            for name, host in targets:
                saved = ('forward' in vars(host), vars(host).get('forward'))
                projection = HNDLParticleProjection(name, host, self.particles, 8, 8,
                    builder=builder, identity=identity)
                self._forwards.append((host, saved, host.forward))
                self.projections[name] = projection
        except BaseException:
            self.close()
            raise

    @contextmanager
    def scaled(self, scale):
        previous = self.scale
        self.scale = scale
        for projection in self.projections.values():
            projection.multiplier = scale
        try:
            yield
        finally:
            self.scale = previous
            for projection in self.projections.values():
                projection.multiplier = previous

    def extra_repr(self):
        return json.dumps(dict(protocol='music3-hndl-routed-qwen3-v1', **self.identity), sort_keys=True)

    def close(self):
        if any(host.forward is not attached for host, _, attached in self._forwards):
            raise RuntimeError('Music3 HNDL forward owner changed; resolve before retrying close')
        for host, (owned, previous), _ in self._forwards:
            if owned:
                host.forward = previous
            else:
                vars(host).pop('forward', None)
        self._forwards.clear()
        self._restore_native()


class Music3HNDLCritic(nn.Module):
    """Same linear/SiLU/linear AR readout through the real HyperGAN builder."""
    score_bound = 0
    def __init__(self, width, condition_width):
        super().__init__()
        builder, identity = verified_builder()
        source = Path(__file__).with_name('critic.hndl').read_text()
        self.graph = builder(source, input_shape={'response': ['B', width], 'condition': ['B', condition_width]},
                             output_shape=['B', 1])
        self.identity = dict(identity, ar_critic_sha256=hashlib.sha256(source.encode()).hexdigest())

    def extra_repr(self):
        return json.dumps(self.identity, sort_keys=True)

    def forward(self, response, condition):
        return self.graph(response=response, condition=condition).squeeze(-1)


class OwnedGame:
    """Keep the cleanup ledger reachable after a forward ownership conflict."""
    def __init__(self, model, slider, adapter, game):
        self.model, self.slider, self.adapter, self.game = model, slider, adapter, game
    def close(self):
        self.adapter.close_checkpointing()
        self.slider.close()
    def __enter__(self):
        return self
    def __exit__(self, *error):
        self.close()


def build_cpu_game(*, width=32, steps=2, seed=7, checkpointing=True, preset='bispf_optimized'):
    """Genuine random Qwen3 fixture only. Reject native width before allocation."""
    verify_core()
    require_audio_width(width, pg_source_profile=PG_SOURCE_PROFILE)
    if width != 32 or type(steps) is not int or not 1 <= steps <= 4 or type(checkpointing) is not bool:
        raise ValueError('bounded candidate requires width32, 1..4 steps and boolean checkpointing')
    if type(seed) is not int or not 0 <= seed <= 2**63 - 101:
        raise ValueError('invalid seed')
    if preset not in ('bispf', 'bispf_optimized'):
        raise ValueError('unsupported PG preset')
    from transformers import Qwen3Config, Qwen3ForCausalLM
    slider = adapter = None
    try:
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
            model = Qwen3ForCausalLM(Qwen3Config(hidden_size=32, intermediate_size=64,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                vocab_size=256, max_position_embeddings=256, attention_dropout=0.,
                use_cache=False, attn_implementation='eager'))
            identity = dict(fixture='random Qwen3; synthetic IDs; HNDL cohort', seed=seed,
                            frozen_base_sha256=_hash(model.state_dict()), width=32)
            slider = Music3HNDLNetwork(model)
            adapter = Music3ARResponseAdapter(model, slider, model_identity=identity,
                checkpoint_modules=tuple(model.model.layers) if checkpointing else (),
                pg_source_profile=PG_SOURCE_PROFILE)
            rows = [adapter.prepare(AudioRow((11, 13, 17), (29, 31, 17), history_tokens=1)),
                    adapter.prepare(AudioRow((23, 37, 41), (53, 59, 41), history_tokens=1))]
            conditions = audio_conditions(rows)
            critic = Music3HNDLCritic(32, conditions.shape[1])
            game = ConditionalPG09(adapter, rows, critic, conditions=conditions, particles=adapter.particles,
                response_domain='continuous_vector', preset=preset, batch_size=2, total_steps=steps,
                seed=seed, pg_source_profile=PG_SOURCE_PROFILE)
        return OwnedGame(model, slider, adapter, game)
    except BaseException:
        if adapter is not None:
            adapter.close_checkpointing()
        if slider is not None:
            slider.close()
        raise
