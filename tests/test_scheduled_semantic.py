from contextlib import ExitStack, contextmanager
import os
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
import torch
from torch import nn

from music3.activation_gate import CORE_SOURCE, SemanticGatePlan
from music3.scheduled_semantic import generate, generate_gated


@pytest.fixture
def native_semantic():
    pytest.importorskip("transformers")
    encoders = pytest.importorskip("diffusers.modular_pipelines.minimax_music3.encoders")
    from diffusers.modular_pipelines.modular_pipeline import PipelineState
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from music3.native_binding import ParticleNetwork

    width = 8
    config = Qwen3Config(vocab_size=encoders._AUDIO_CODE_OFFSET + encoders._SEMANTIC_VOCAB_SIZE,
                         hidden_size=width, intermediate_size=16, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=2, head_dim=4)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(26)
        language_model = Qwen3ForCausalLM(config).eval()
        slider = ParticleNetwork(language_model).eval()
        with torch.no_grad():
            for layer in slider.network.unet_loras:
                layer.lora_up.weight.normal_(std=0.15)

    class RVQ(nn.Module):
        config = SimpleNamespace(num_codebooks=3, audio_vocab_size=64)

        def __init__(self):
            super().__init__()
            self.projection = nn.Identity()
            self.audio_embeddings = nn.Embedding(
                (self.config.num_codebooks - 1) * self.config.audio_vocab_size, width)
            self.audio_heads = nn.ModuleList([
                nn.Linear(width, self.config.audio_vocab_size)
                for _ in range(self.config.num_codebooks - 1)
            ])
            self.depth_forwards = 0

        def forward(self, hidden):
            self.depth_forwards += 1
            return hidden

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(27)
        rvq = RVQ().eval()
    block = encoders.MiniMaxMusic3AutoregressiveStep()

    class Pipeline:
        frame_rate = 25.0
        eos_prediction = None
        fail = False

        def __init__(self):
            self.language_model = language_model
            self.rvq_depth_decoder = rvq
            self.applied = []
            self.feedback = []

        def __call__(self, *, audio_duration):
            self.predictions = 0
            self.rvq_depth_decoder.depth_forwards = 0
            state = PipelineState(values={
                "text_ids": torch.tensor([[1, 2, 3], [1, 4, 3]]),
                "audio_duration": audio_duration,
                "generator": torch.Generator().manual_seed(13),
            })
            _, result = block(self, state)
            return result.get("frame_hiddens")

        @property
        def active(self):
            return slider.network.lora_scale

        @contextmanager
        def scale(self, value):
            self.applied.append(value)
            with slider.scaled(value):
                yield

    pipe = Pipeline()

    def fail_on_edit(_layer, _args, _output):
        if pipe.fail and pipe.active:
            raise RuntimeError("fixture failure inside the scoped adapted LM")

    def schedule_eos(_module, _args, logits):
        if pipe.predictions == pipe.eos_prediction:
            logits.fill_(-1e4)
            logits[..., encoders._AUDIO_END_TOKEN_ID] = 1e4
        else:
            logits[..., encoders._AUDIO_END_TOKEN_ID] = -1e4
        pipe.predictions += 1
        return logits

    def observe_input(_model, _args, kwargs):
        if kwargs.get("past_key_values") is not None:
            pipe.feedback.append(kwargs["inputs_embeds"][0, 0].clone())

    with ExitStack() as handles:
        handles.callback(language_model.model.register_forward_hook(fail_on_edit).remove)
        handles.callback(language_model.model.register_forward_pre_hook(
            observe_input, with_kwargs=True).remove)
        handles.callback(language_model.lm_head.register_forward_hook(schedule_eos).remove)
        yield pipe, width


@pytest.fixture
def core_wheel():
    wheel = os.environ.get("MUSIC3_CORE_269_WHEEL")
    if not wheel:
        pytest.skip("Provide MUSIC3_CORE_269_WHEEL for the optional verified steering module")
    return Path(wheel)


def test_native_semantic_loop_preserves_constant_cfg_and_applies_frame_strengths(native_semantic):
    pipe, width = native_semantic
    with pipe.scale(0.5):
        baseline = pipe(audio_duration=0.12)
    pipe.applied.clear()
    constant, applied = generate(pipe, scale_adapter=pipe.scale, audio_duration=0.12,
                                 strength=0.5, step_gains=[1] * 4)
    assert applied == pipe.applied == [0.5] * 4
    assert torch.equal(constant, baseline) and baseline.shape == (1, 3, 3 * width)
    pipe.applied.clear()
    varied, applied = generate(pipe, scale_adapter=pipe.scale, audio_duration=0.12,
                               strength=0.5, step_gains=[1.5, 0.5, 1, 1])
    assert applied == pipe.applied == [0.75, 0.25, 0.5, 0.5]
    assert not torch.equal(varied, baseline)
    assert pipe.active == 0


@pytest.mark.parametrize("method", ["cfp", "clabs", "random"])
def test_gated_native_replay_with_early_eos(native_semantic, core_wheel, method):
    pipe, width = native_semantic
    pipe.eos_prediction = 3
    with pipe.scale(0):
        baseline = pipe(audio_duration=0.20)
    direction = torch.eye(width)[0]
    scores = [float(x[0]) for x in pipe.feedback]
    assert len(scores) == 3
    low, high = min(scores), max(scores)
    assert low < high
    midpoint = (low + high) / 2
    options = {"direction": direction}
    parameter = midpoint
    if method == "random":
        options = {"seed": 3}
        parameter = 0.5
    elif method == "clabs":
        options.update(seed=3, clabs=dict(mean=midpoint, std=high - low, epsilon=1e-6,
                                         setpoint=0, proportional_gain=1, integral_gain=0.2, beta=0.2))
        parameter = 0.5
    plan = SemanticGatePlan(core_wheel=core_wheel, method=method, parameter=parameter,
                            calibration_sha256="0" * 64, **options)
    pipe.feedback.clear()
    pipe.applied.clear()

    result, receipt = generate_gated(pipe, scale_adapter=pipe.scale, audio_duration=0.20,
                                    gate_plan=plan, strength=0.5)
    decisions = receipt["decisions"]
    assert receipt["applied_strengths"] == pipe.applied
    assert receipt["applied_strengths"][0] == 0
    assert receipt["eligible"] == len(decisions) == 3
    assert receipt["fired"] == sum(entry["fired"] for entry in decisions)
    assert receipt["gate_rate"] == receipt["fired"] / receipt["eligible"]
    assert receipt["total_strength"] == 0.5 * receipt["fired"]
    assert [entry["frame_prediction"] for entry in decisions] == [0, 1, 2]
    assert result.shape == baseline.shape == (1, 2, 3 * width)
    # RVQ runs twice for each of the warm-up and two emitted frames, and never on EOS.
    assert pipe.rvq_depth_decoder.depth_forwards == 6
    reference = plan._new_gate()
    for entry, embedding in zip(decisions, pipe.feedback, strict=True):
        score = None if method == "random" else float(embedding[0])
        expected = reference.step(score)
        assert entry["projection"] == score
        assert entry["fired"] == expected.fired
        assert entry["probability"] == expected.probability
    if method == "cfp":
        assert 0 < receipt["fired"] < receipt["eligible"]

    gains = [value / 0.5 for value in receipt["applied_strengths"]] + [0, 0]
    replay, applied = generate(pipe, scale_adapter=pipe.scale, audio_duration=0.20,
                               strength=0.5, step_gains=gains)
    assert applied == receipt["applied_strengths"]
    assert torch.equal(result, replay)
    repeated, repeat_receipt = generate_gated(pipe, scale_adapter=pipe.scale, audio_duration=0.20,
                                            gate_plan=plan, strength=0.5)
    assert repeat_receipt == receipt and torch.equal(repeated, result)
    assert pipe.active == 0


def test_gated_off_fixed_and_exception_cleanup(native_semantic, core_wheel):
    pipe, _ = native_semantic
    off = SemanticGatePlan(core_wheel=core_wheel, method="random", parameter=0,
                           calibration_sha256="0" * 64, seed=3)
    on = SemanticGatePlan(core_wheel=core_wheel, method="random", parameter=1,
                          calibration_sha256="0" * 64, seed=3)
    with pipe.scale(0):
        baseline = pipe(audio_duration=0.12)
    result, receipt = generate_gated(pipe, scale_adapter=pipe.scale, audio_duration=0.12,
                                    gate_plan=off, strength=0.5)
    assert receipt["applied_strengths"] == [0] * 4
    assert torch.equal(result, baseline)
    expected, _ = generate(pipe, scale_adapter=pipe.scale, audio_duration=0.12,
                           strength=0.5, step_gains=[0, 1, 1, 1])
    result, receipt = generate_gated(pipe, scale_adapter=pipe.scale, audio_duration=0.12,
                                    gate_plan=on, strength=0.5)
    assert receipt["applied_strengths"] == [0, 0.5, 0.5, 0.5]
    assert torch.equal(result, expected)

    pipe.fail = True
    with pytest.raises(RuntimeError, match="fixture failure"):
        generate_gated(pipe, scale_adapter=pipe.scale, audio_duration=0.12,
                       gate_plan=on, strength=0.5)
    assert pipe.active == 0
    starts = len(pipe.applied)
    pipe.fail = False
    assert torch.equal(pipe(audio_duration=0.12), baseline)
    assert len(pipe.applied) == starts


def test_changed_steering_wheel_is_rejected(core_wheel, tmp_path):
    from music3.activation_gate import load_steering_core

    load_steering_core(core_wheel)
    with ZipFile(core_wheel) as original:
        source = original.read(CORE_SOURCE)
    tampered = tmp_path / "changed.whl"
    with ZipFile(tampered, "w") as changed:
        changed.writestr(CORE_SOURCE, b" " + source[1:])
    with pytest.raises(ValueError, match="core PR #269"):
        load_steering_core(tampered)
