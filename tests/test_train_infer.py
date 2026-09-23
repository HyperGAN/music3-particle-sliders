"""CPU train/infer smoke. No Hub download."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from music3.defaults import BASE_MODEL_ID, HUB_WEIGHTS_ID
from music3.infer import infer, parse_args as parse_infer
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
    with pytest.raises(RuntimeError, match="does not load"):
        train(parse_args(["--live", "--model_dir", str(checkout), "--steps", "1"]))


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
