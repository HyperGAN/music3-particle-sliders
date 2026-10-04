"""CPU train/infer smoke. No Hub download."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import torch
from torch import nn

from music3 import live as live_mod
from music3.attach import attach_stamp_bridge, bind_live_host
from music3.defaults import BASE_MODEL_ID, HUB_WEIGHTS_ID
from music3.infer import infer, parse_args as parse_infer
from music3.live import LoadedHost, run_live
from music3.prompts import load_prompts
from music3.surface import declare, load_stamp
from music3.train import parse_args, train

ROOT = Path(__file__).resolve().parents[1]


def _run(script: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "script",
    [
        "train_music3.py",
        "train_lm_slider_music3.py",
        "train_lora_music3.py",
        "train_lora_music3_particle.py",
        "train_encoder_music3.py",
        "infer_music3.py",
    ],
)
def test_help(script: str):
    proc = _run(script, ["--help"])
    assert proc.returncode == 0, proc.stderr
    assert BASE_MODEL_ID in proc.stdout
    assert "--dummy" in proc.stdout


def test_dummy_train_calls_the_stamp(tmp_path: Path):
    sidecar_path = train(
        parse_args(
            [
                "--dummy",
                "--name",
                "energy-cpu",
                "--save_dir",
                str(tmp_path),
                "--steps",
                "8",
                "--seed",
                "7",
                "--g_lr",
                "0.0002",
            ]
        )
    )
    payload = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
    assert payload["require"] == "ok"
    assert payload["formulation_id"] == "particle-gmix-1600-v2"
    assert payload["architecture_id"] == "gmix"
    assert payload["lm_target"] == "faithful_plus_neu"
    assert payload["model_id"] == BASE_MODEL_ID
    assert payload["hub_weights"] == HUB_WEIGHTS_ID
    assert payload["host"] == "lm"
    assert payload["target_replace"] == ["Qwen3Attention"]
    assert payload["bridge_class"] == "RoutedMLP"
    assert payload["bridge_module"] == "particle_sliders.reference"
    assert payload["critic_class"] == "GlobalMixErrorCritic"
    assert payload["critic_module"] == "particle_sliders.reference"
    assert payload["regularizer_module"].startswith("particlegan")
    assert payload["dummy"] is True
    assert payload["steps_ran"] == 4
    assert payload["last"]["g_lr"] == pytest.approx(0.0002)
    lines = [
        json.loads(line)
        for line in (tmp_path / "energy-cpu_train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["step"] for row in lines] == [1, 2, 3, 4]
    assert all(row["sigma"] > 0 for row in lines)
    assert lines[0]["d_penalty"] == 0.0
    assert "d_penalty" in lines[3]


def test_transformer_and_encoder_hosts(tmp_path: Path):
    transformer = json.loads(
        Path(
            train(
                parse_args(
                    ["--dummy", "--save_dir", str(tmp_path / "tf"), "--steps", "1", "--targets", "attn"],
                    host="transformer",
                )
            )
        ).read_text(encoding="utf-8")
    )
    assert transformer["host"] == "transformer"
    assert transformer["target_replace"] == ["MiniMaxMusic3Attention"]
    encoder = json.loads(
        Path(
            train(
                parse_args(["--dummy", "--save_dir", str(tmp_path / "enc"), "--steps", "1"], host="encoder")
            )
        ).read_text(encoding="utf-8")
    )
    assert encoder["target_replace"] == ["MiniMaxMusic3ConditionEncoder"]
    particle = json.loads(
        Path(
            train(
                parse_args(
                    ["--dummy", "--save_dir", str(tmp_path / "particle"), "--steps", "1"],
                    host="particle",
                )
            )
        ).read_text(encoding="utf-8")
    )
    assert particle["entry_host"] == "particle"
    assert particle["host"] == "lm"
    assert particle["target_replace"] == ["Qwen3Attention"]


def test_forked_flags_are_rejected():
    with pytest.raises(SystemExit):
        train(parse_args(["--dummy", "--lm_target", "v9", "--steps", "1"]))
    with pytest.raises(SystemExit):
        train(parse_args(["--dummy", "--loss", "nmse", "--steps", "1"]))
    with pytest.raises(SystemExit):
        train(parse_args(["--dummy", "--rank", "64", "--steps", "1"]))
    with pytest.raises(SystemExit):
        train(parse_args(["--dummy", "--model_id", "krea/Krea-2-Raw", "--steps", "1"]))


def test_one_row_card_cannot_train():
    with pytest.raises(ValueError, match="at least two"):
        train(
            parse_args(
                [
                    "--dummy",
                    "--prompts_file",
                    "configs/music3/prompts-music3.yaml",
                    "--steps",
                    "1",
                ]
            )
        )


def test_live_does_not_download(tmp_path: Path):
    with pytest.raises(RuntimeError, match="does not download"):
        train(parse_args(["--live", "--steps", "1"]))
    missing = tmp_path / "missing-music3"
    with pytest.raises(FileNotFoundError):
        train(parse_args(["--live", "--model_dir", str(missing), "--steps", "1"]))
    checkout = tmp_path / "MiniMax-Music-3"
    checkout.mkdir()
    with pytest.raises(RuntimeError, match="language_model") as caught:
        train(parse_args(["--live", "--model_dir", str(checkout), "--steps", "1"]))
    message = str(caught.value)
    assert "does not download" in message
    assert "does not load the Music 3 weights" not in message
    language_model = checkout / "language_model"
    language_model.mkdir()
    (language_model / "config.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="weight"):
        train(parse_args(["--live", "--model_dir", str(checkout), "--steps", "1"]))
    (language_model / "model.safetensors").write_bytes(b"not-a-shard")
    with pytest.raises(RuntimeError, match="tokenizer"):
        train(parse_args(["--live", "--model_dir", str(checkout), "--steps", "1"]))
    with pytest.raises(RuntimeError, match="transformer/config.json"):
        train(
            parse_args(
                ["--live", "--model_dir", str(checkout), "--steps", "1"],
                host="transformer",
            )
        )
    with pytest.raises(RuntimeError, match="condition_encoder"):
        train(
            parse_args(
                ["--live", "--model_dir", str(checkout), "--steps", "1"],
                host="encoder",
            )
        )


def test_print_card_lists_the_release():
    card = train(parse_args(["--print_card"]))
    assert card["formulation_id"] == "particle-gmix-1600-v2"
    assert card["base_model"] == BASE_MODEL_ID
    assert len(card["controls"]) == 16
    assert card["controls"][0]["id"] == "female"


def test_dummy_infer_writes_a_listen_card(tmp_path: Path):
    path = infer(
        parse_infer(
            [
                "--dummy",
                "--name",
                "energy-listen",
                "--out_dir",
                str(tmp_path),
                "--scales=-2,-1,0,1,2",
                "--prompts_file",
                "configs/music3/prompts-music3.yaml",
            ]
        )
    )
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert payload["rendered_audio"] is False
    assert payload["scales"] == [-2.0, -1.0, 0.0, 1.0, 2.0]
    assert "BPM: 110" in payload["neutral"]
    assert payload["formulation_id"] == "particle-gmix-1600-v2"
    assert payload["hub_weights"] == HUB_WEIGHTS_ID
    assert not any(tmp_path.glob("*.wav"))


def test_infer_without_dummy_refuses():
    with pytest.raises(RuntimeError, match="--dummy"):
        infer(parse_infer(["--name", "nope"]))


def test_attach_binds_stamp_bridge_to_host_linears():
    """Stand-in modules with Music 3 class names. No checkpoint is written."""
    stamp = load_stamp()
    hidden = 4
    root = MiniMaxMusic3Transformer1DModel(hidden)
    attention_only = attach_stamp_bridge(root, stamp, ["MiniMaxMusic3Attention"])
    attention_names = {bridge.name for bridge in attention_only}
    assert attention_names == {
        "blocks.0.attn.to_q",
        "blocks.0.attn.to_k",
        "blocks.0.attn.to_v",
        "blocks.0.attn.to_out.0",
    }
    full = MiniMaxMusic3Transformer1DModel(hidden)
    bound = bind_live_host(
        full,
        stamp,
        [
            "MiniMaxMusic3Attention",
            "MiniMaxMusic3TransformerBlock",
            "MiniMaxMusic3Transformer1DModel",
        ],
    )
    names = {bridge.name for bridge in bound}
    assert names == attention_names | {"blocks.0.ff", "proj_in"}
    assert all(type(bridge.bridge).__name__ == "RoutedMLP" for bridge in bound)
    assert all(type(bridge.bridge).__module__ == "particle_sliders.reference" for bridge in bound)
    shared = bound[0]._particles.param
    assert all(bridge._particles.param is shared for bridge in bound)
    assert not any(bridge.name.endswith("conv") for bridge in bound)

    linear = full.blocks[0].attn.to_q
    bridge = linear._music3_stamp_bridge
    assert linear.forward.__func__ is bridge.forward.__func__
    probe = torch.randn(2, 3, hidden)
    reference = torch.nn.functional.linear(probe, linear.weight, linear.bias)
    bridge.scale = 0.0
    assert torch.allclose(linear(probe), reference)
    bridge.scale = 1.0
    with torch.no_grad():
        bridge.up.weight.add_(0.05)
    shifted = linear(probe)
    assert not torch.allclose(shifted, reference)
    shifted.sum().backward()
    assert bridge.up.weight.grad is not None
    assert any(param.grad is not None and float(param.grad.abs().sum()) > 0 for param in bridge.bridge.parameters())

    language = _TinyLM(hidden)
    lm_bound = bind_live_host(language, stamp, ["Qwen3Attention"])
    lm_names = {bridge.name for bridge in lm_bound}
    assert lm_names == {
        "inner.attn.q_proj",
        "inner.attn.k_proj",
        "inner.attn.v_proj",
        "inner.attn.o_proj",
    }
    assert not hasattr(language.lm_head, "_music3_stamp_bridge")

    encoder = MiniMaxMusic3ConditionEncoder(hidden)
    encoder_bound = bind_live_host(encoder, stamp, ["MiniMaxMusic3ConditionEncoder"])
    assert [bridge.name for bridge in encoder_bound] == ["proj"]
    assert not hasattr(encoder.norm, "_music3_stamp_bridge")


@pytest.mark.parametrize(
    ("host_name", "linear_path"),
    [
        ("lm", ("inner", "attn", "q_proj")),
        ("transformer", ("blocks", 0, "attn", "to_q")),
        ("encoder", ("proj",)),
    ],
)
def test_live_entry_binds_stamp_bridge_before_stepping(tmp_path: Path, monkeypatch, host_name: str, linear_path):
    standin = {
        "lm": _TinyLM(4),
        "transformer": MiniMaxMusic3Transformer1DModel(4),
        "encoder": MiniMaxMusic3ConditionEncoder(4),
    }[host_name]
    checkout = tmp_path / "MiniMax-Music-3"
    checkout.mkdir()

    def load(model_dir, host):
        assert Path(model_dir) == checkout
        assert host["name"] == host_name
        return LoadedHost(
            model=standin,
            tokenizer=None,
            kind=host["kind"],
            model_dir=Path(model_dir),
            subfolder="stand-in",
            host_name=host_name,
        )

    def stop_after_bind(stamp, declared, loaded, bound, **kwargs):
        assert bound
        assert loaded.model is standin
        assert {type(bridge.bridge).__module__ for bridge in bound} == {"particle_sliders.reference"}
        assert {type(bridge.bridge).__name__ for bridge in bound} == {"RoutedMLP"}
        raise RuntimeError("stop-after-bind")

    monkeypatch.setattr(live_mod, "load_live_host", load)
    monkeypatch.setattr(live_mod, "run_live", stop_after_bind)
    with pytest.raises(RuntimeError, match="stop-after-bind"):
        train(
            parse_args(
                ["--live", "--model_dir", str(checkout), "--steps", "1", "--save_dir", str(tmp_path / "out")],
                host=host_name,
            )
        )
    linear = standin
    for part in linear_path:
        linear = linear[part] if isinstance(part, int) else getattr(linear, part)
    assert type(linear._music3_stamp_bridge.bridge).__module__ == "particle_sliders.reference"
    assert not (tmp_path / "out").exists()


def test_live_lm_stamp_step_updates_the_bound_bridge(monkeypatch):
    """Same attach and stamp step as --live, on a tiny stand-in.

    The stand-in is not MiniMax-Music-3 and this test does not write a slider.
    """
    monkeypatch.setattr(
        live_mod,
        "assemble_music3_prompt",
        lambda prompt, lyrics: f"{prompt}\n{lyrics}\n<|audio_start|>",
    )
    model = _TinyLM(8)
    stamp = load_stamp()
    declared = declare(stamp)
    bound = bind_live_host(model, stamp, ["Qwen3Attention"])
    tracked = model.inner.attn.q_proj._music3_stamp_bridge
    before = tracked.up.weight.detach().clone()
    rows, _meta = load_prompts(ROOT / "configs/music3/prompts-particle.yaml")
    loaded = LoadedHost(
        model=model,
        tokenizer=_StandInTokenizer(),
        kind="language_model",
        model_dir=ROOT,
        subfolder="language_model",
        host_name="lm",
    )
    logs, built = run_live(stamp, declared, loaded, bound, prompts=rows, steps=1, seed=7)
    assert logs[0]["step"] == 1
    assert "g_loss" in logs[0]
    assert not torch.equal(before, tracked.up.weight.detach())
    assert type(tracked.bridge).__module__ == "particle_sliders.reference"
    assert type(built["bridge"]).__module__ == "particle_sliders.reference"
    assert type(built["critic"]).__module__ == "particle_sliders.reference"


def test_transformer_live_step_names_missing_latents():
    """The module forward is not in conceptmod. Missing local inputs fail closed.

    The stand-in below is not a Music 3 weight load.
    """
    model = MiniMaxMusic3Transformer1DModel(4)
    stamp = load_stamp()
    declared = declare(stamp)
    bound = bind_live_host(
        model,
        stamp,
        ["MiniMaxMusic3Attention", "MiniMaxMusic3TransformerBlock", "MiniMaxMusic3Transformer1DModel"],
    )
    rows, _meta = load_prompts(ROOT / "configs/music3/prompts-particle.yaml")
    loaded = LoadedHost(
        model=model,
        tokenizer=None,
        kind="transformer",
        model_dir=ROOT,
        subfolder="transformer",
        host_name="transformer",
    )
    with pytest.raises(RuntimeError, match="MiniMaxMusic3Transformer1DModel.forward") as caught:
        run_live(stamp, declared, loaded, bound, prompts=rows, steps=1, seed=1)
    message = str(caught.value)
    assert "stamp.bridge()" in message
    assert "blocks.0.attn.to_q" in message
    assert "latents.pt" in message
    assert "does not copy" in message
    assert "conceptmod/textsliders/train_lora_music3.py" in message
    assert "stays in" not in message
    assert "does not invent latents" in message
    assert type(model.blocks[0].attn.to_q._music3_stamp_bridge.bridge).__module__ == "particle_sliders.reference"


def test_encoder_live_step_names_missing_frame_hiddens():
    """Same hole as the transformer: call the module forward, or name the missing tensors."""
    model = MiniMaxMusic3ConditionEncoder(4)
    stamp = load_stamp()
    declared = declare(stamp)
    bound = bind_live_host(model, stamp, ["MiniMaxMusic3ConditionEncoder"])
    rows, _meta = load_prompts(ROOT / "configs/music3/prompts-particle.yaml")
    loaded = LoadedHost(
        model=model,
        tokenizer=None,
        kind="condition_encoder",
        model_dir=ROOT,
        subfolder="condition_encoder",
        host_name="encoder",
    )
    with pytest.raises(RuntimeError, match="MiniMaxMusic3ConditionEncoder.forward") as caught:
        run_live(stamp, declared, loaded, bound, prompts=rows, steps=1, seed=1)
    message = str(caught.value)
    assert "neutral.pt" in message
    assert "cache_real_hiddens" in message
    assert "does not copy" in message
    assert "stays in" not in message


def _host_linear_names(root: nn.Module, classes: list[str]) -> set[str]:
    wanted = set(classes)
    names: set[str] = set()
    seen: set[int] = set()
    for name, module in root.named_modules():
        if module.__class__.__name__ not in wanted:
            continue
        for child_name, child in module.named_modules():
            if not isinstance(child, nn.Linear) or id(child) in seen:
                continue
            seen.add(id(child))
            names.add(".".join(part for part in (name, child_name) if part))
    return names


def test_real_host_classes_bind_nn_linear(tmp_path: Path):
    """Construct the published host classes with no checkpoint.

    Random initialization is not a MiniMax-Music-3 weight load and this test
    does not write a slider. ``--dummy`` is not this path.
    """
    try:
        from diffusers.models.condition_embedders.condition_embedder_minimax_music3 import (
            MiniMaxMusic3ConditionEncoder as RealEncoder,
        )
        from diffusers.models.transformers.transformer_minimax_music3 import (
            MiniMaxMusic3Transformer1DModel as RealTransformer,
        )
        from transformers import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention as RealQwen
    except ImportError as exc:
        pytest.skip(
            "diffusers/transformers Music 3 host classes are not installed "
            f"({exc}). They are not defined in this repo. This test does not "
            "download a MiniMax-Music-3 checkpoint."
        )
    try:
        transformer = RealTransformer(
            in_channels=4,
            condition_dim=8,
            num_layers=1,
            num_attention_heads=2,
            attention_head_dim=4,
            ff_inner_dim=8,
            rotary_dim=4,
            fourier_embedding_dim=8,
        )
        encoder = RealEncoder(condition_hidden_dim=6, num_condition_layers=2, out_dim=4)
        attention = RealQwen(
            Qwen3Config(
                hidden_size=16,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=8,
                intermediate_size=32,
                num_hidden_layers=1,
            ),
            layer_idx=0,
        )
    except Exception as exc:
        pytest.skip(
            "Published Music 3 host classes could not be constructed without a "
            f"checkpoint ({type(exc).__name__}: {exc}). This test does not "
            "download MiniMax-Music-3 weights."
        )

    stamp = load_stamp()
    classes = [
        "MiniMaxMusic3Attention",
        "MiniMaxMusic3TransformerBlock",
        "MiniMaxMusic3Transformer1DModel",
    ]
    transformer.requires_grad_(False)
    transformer.eval()
    expected = _host_linear_names(transformer, classes)
    bound = bind_live_host(transformer, stamp, classes)
    assert {bridge.name for bridge in bound} == expected
    assert all("_music3_stamp_bridge" not in bridge.name for bridge in bound)
    assert "transformer_blocks.0.attn.to_q" in {bridge.name for bridge in bound}
    assert "proj_in" in {bridge.name for bridge in bound}
    assert all(type(bridge.bridge).__name__ == "RoutedMLP" for bridge in bound)
    assert all(type(bridge.bridge).__module__ == "particle_sliders.reference" for bridge in bound)
    tracked = transformer.transformer_blocks[0].attn.to_q._music3_stamp_bridge
    assert transformer.transformer_blocks[0].attn.to_q.forward.__func__ is tracked.forward.__func__

    from music3.surface_forward import encoder_forward, transformer_forward

    latents = torch.randn(2, 4, 5)
    step = torch.full((2,), 0.4)
    cond = torch.randn(2, 5, 8)
    transformer._music3_stamp_bridges.set_scale(0.0)
    velocity = transformer_forward(transformer, latents, step, cond)
    assert tuple(velocity.shape) == (2, 4, 5)
    transformer._music3_stamp_bridges.set_scale(1.0)
    with torch.no_grad():
        tracked.up.weight.add_(0.05)
    shifted = transformer_forward(transformer, latents, step, cond)
    assert not torch.allclose(shifted, velocity)
    shifted.float().sum().backward()
    assert any(
        param.grad is not None and float(param.grad.abs().sum()) > 0 for param in tracked.bridge.parameters()
    )

    rows, _meta = load_prompts(ROOT / "configs/music3/prompts-particle.yaml")
    surface = tmp_path / "surface" / "transformer"
    surface.mkdir(parents=True)
    neutral = torch.randn(len(rows), 5, 8)
    torch.save(torch.randn(len(rows), 4, 5), surface / "latents.pt")
    torch.save(neutral, surface / "neutral.pt")
    torch.save(neutral + 1, surface / "positive.pt")
    fresh = RealTransformer(
        in_channels=4,
        condition_dim=8,
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=4,
        ff_inner_dim=8,
        rotary_dim=4,
        fourier_embedding_dim=8,
    )
    fresh.requires_grad_(False)
    fresh.eval()
    fresh_bound = bind_live_host(fresh, stamp, classes)
    calls = {"n": 0}
    original = fresh.forward

    def _count_forward(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    fresh.forward = _count_forward
    before = fresh.transformer_blocks[0].attn.to_q._music3_stamp_bridge.up.weight.detach().clone()
    loaded = LoadedHost(
        model=fresh,
        tokenizer=None,
        kind="transformer",
        model_dir=tmp_path,
        subfolder="transformer",
        host_name="transformer",
    )
    logs, built = run_live(stamp, declare(stamp), loaded, fresh_bound, prompts=rows, steps=1, seed=7)
    assert calls["n"] >= 3
    assert logs[0]["step"] == 1
    assert "g_loss" in logs[0]
    assert not torch.equal(before, fresh.transformer_blocks[0].attn.to_q._music3_stamp_bridge.up.weight.detach())
    assert type(built["bridge"]).__module__ == "particle_sliders.reference"
    assert not any(tmp_path.glob("*_last.json"))
    assert not any(tmp_path.glob("*_stamp_bridge.pt"))

    with pytest.raises(RuntimeError, match="no nn.Linear") as caught:
        bind_live_host(encoder, stamp, ["MiniMaxMusic3ConditionEncoder"])
    assert "Conv1d" in str(caught.value)
    assert isinstance(encoder.proj, nn.Conv1d)
    encoded = encoder_forward(encoder, torch.randn(2, 4, 12))
    assert encoded.ndim == 3 and encoded.shape[0] == 2

    expected_lm = _host_linear_names(attention, ["Qwen3Attention"])
    lm_bound = bind_live_host(attention, stamp, ["Qwen3Attention"])
    assert {bridge.name for bridge in lm_bound} == expected_lm
    assert {"q_proj", "k_proj", "v_proj", "o_proj"} <= {bridge.name for bridge in lm_bound}
    assert all(type(bridge.bridge).__module__ == "particle_sliders.reference" for bridge in lm_bound)


class Qwen3Attention(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)


class _Inner(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.embed = nn.Embedding(64, hidden)
        self.attn = Qwen3Attention(hidden)

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embed(input_ids.clamp(0, 63))
        mixed = hidden_states + hidden_states.mean(dim=1, keepdim=True)
        return self.attn.o_proj(self.attn.q_proj(mixed))


class _TinyLM(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.inner = _Inner(hidden)
        self.lm_head = nn.Linear(hidden, 4, bias=False)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, use_cache=False):
        hidden_states = self.inner(input_ids, attention_mask)
        from types import SimpleNamespace

        return SimpleNamespace(hidden_states=(hidden_states,) if output_hidden_states else None)


class MiniMaxMusic3Attention(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.to_q = nn.Linear(hidden, hidden)
        self.to_k = nn.Linear(hidden, hidden)
        self.to_v = nn.Linear(hidden, hidden)
        self.to_out = nn.ModuleList([nn.Linear(hidden, hidden)])
        self.conv = nn.Conv1d(hidden, hidden, 1)


class MiniMaxMusic3TransformerBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.attn = MiniMaxMusic3Attention(hidden)
        self.ff = nn.Linear(hidden, hidden)


class MiniMaxMusic3Transformer1DModel(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.blocks = nn.ModuleList([MiniMaxMusic3TransformerBlock(hidden)])
        self.proj_in = nn.Linear(hidden, hidden)


class MiniMaxMusic3ConditionEncoder(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)


class _StandInTokenizer:
    unk_token_id = 0

    def convert_tokens_to_ids(self, token: str) -> int:
        if token == "<|audio_start|>":
            return 7
        return 1

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        digest = 0
        for char in text:
            digest = (digest * 131 + ord(char)) % 997
        first = (digest % 40) + 1
        ids = torch.tensor([[first, (first * 3) % 40 + 1, 7]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
