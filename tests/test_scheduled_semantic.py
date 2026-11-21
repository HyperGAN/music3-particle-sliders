from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from music3.scheduled_semantic import generate


def test_native_semantic_loop_preserves_constant_cfg_and_applies_frame_strengths():
    pytest.importorskip("transformers")
    encoders = pytest.importorskip("diffusers.modular_pipelines.minimax_music3.encoders")
    from diffusers.modular_pipelines.modular_pipeline import PipelineState
    from transformers import Qwen3Config, Qwen3ForCausalLM

    width = 8
    config = Qwen3Config(vocab_size=encoders._AUDIO_CODE_OFFSET + encoders._SEMANTIC_VOCAB_SIZE,
                         hidden_size=width, intermediate_size=16, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=2, head_dim=4)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(26)
        language_model = Qwen3ForCausalLM(config).eval()

    class RVQ(nn.Module):
        config = SimpleNamespace(num_codebooks=2, audio_vocab_size=64)

        def __init__(self):
            super().__init__()
            self.projection = nn.Identity()
            self.audio_embeddings = nn.Embedding(self.config.audio_vocab_size, width)
            self.audio_heads = nn.ModuleList([nn.Linear(width, self.config.audio_vocab_size)])

        def forward(self, hidden):
            return hidden

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(27)
        rvq = RVQ().eval()

    block = encoders.MiniMaxMusic3AutoregressiveStep()
    class Pipeline:
        frame_rate = 25.0
        rvq_depth_decoder = rvq

        def __call__(self, *, audio_duration):
            state = PipelineState(values={
                "text_ids": torch.tensor([[1, 2, 3], [1, 4, 3]]),
                "audio_duration": audio_duration,
                "generator": torch.Generator().manual_seed(13),
            })
            _, result = block(self, state)
            return result.get("frame_hiddens")

    pipe = Pipeline()
    pipe.language_model = language_model
    active = 0.0
    observed = []

    @contextmanager
    def scaled(value):
        nonlocal active
        previous, active = active, value
        observed.append(value)
        try:
            yield
        finally:
            active = previous

    def add_delta(_layer, _args, output):
        output.last_hidden_state = output.last_hidden_state + active * 0.1
        return output

    def suppress_eos(_module, _args, logits):
        logits[..., encoders._AUDIO_END_TOKEN_ID] = -1e4
        return logits

    with contextmanager_hooks(language_model, add_delta, suppress_eos):
        with scaled(0.5):
            baseline = pipe(audio_duration=0.12)
        observed.clear()
        constant, applied = generate(pipe, scale_adapter=scaled, audio_duration=0.12,
                                     strength=0.5, step_gains=[1] * 4)
        assert applied == observed == [0.5] * 4
        assert torch.equal(constant, baseline) and baseline.shape == (1, 3, 2 * width)
        observed.clear()
        varied, applied = generate(pipe, scale_adapter=scaled, audio_duration=0.12,
                                   strength=0.5, step_gains=[1.5, 0.5, 1, 1])
        assert applied == observed == [0.75, 0.25, 0.5, 0.5]
        assert not torch.equal(varied, baseline)
        assert active == 0


@contextmanager
def contextmanager_hooks(model, delta, eos):
    delta_handle = model.model.register_forward_hook(delta)
    eos_handle = model.lm_head.register_forward_hook(eos)
    try:
        yield
    finally:
        delta_handle.remove()
        eos_handle.remove()
