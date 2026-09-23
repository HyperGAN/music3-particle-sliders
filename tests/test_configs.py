"""Prompt cards, host names, and the published Hub release."""

from __future__ import annotations

from pathlib import Path

import yaml

from music3.cards import control_paths, load_sample_cards, release_card
from music3.defaults import BASE_MODEL_ID, HUB_WEIGHTS_ID
from music3.hosts import resolve_host, target_replace
from music3.prompts import load_prompts

ROOT = Path(__file__).resolve().parents[1]


def test_energy_prompt_is_the_historical_card():
    rows, meta = load_prompts(ROOT / "configs/music3/prompts-music3.yaml")
    assert len(rows) == 1
    assert "BPM: 140" in rows[0].positive
    assert "BPM: 70" in rows[0].negative
    assert rows[0].neutral != rows[0].positive
    assert meta.recommended_range == (0.0, 1.0)
    text = (ROOT / "configs/music3/prompts-music3.yaml").read_text(encoding="utf-8")
    assert "Louder now or fade away" in text


def test_particle_card_has_two_unipolar_rows():
    rows, meta = load_prompts(ROOT / "configs/music3/prompts-particle.yaml")
    assert len(rows) >= 2
    assert meta.concept == "energy"
    assert meta.plus_label == "loud"
    assert all(row.lyrics for row in rows)


def test_configs_point_at_this_repo_and_the_base_id():
    for name in ("config-music3.yaml", "config-music3-encoder.yaml"):
        raw = yaml.safe_load((ROOT / "configs/music3" / name).read_text(encoding="utf-8"))
        assert raw["pretrained_model"]["name_or_path"] == BASE_MODEL_ID
        assert raw["prompts_file"] == "configs/music3/prompts-music3.yaml"
        assert raw["network"]["rank"] == 8
        assert (ROOT / raw["prompts_file"]).exists()


def test_sample_cards_match_the_september_release():
    catalog = load_sample_cards()
    assert catalog["hub_weights"] == HUB_WEIGHTS_ID
    assert catalog["base_model"] == BASE_MODEL_ID
    ids = [item["id"] for item in catalog["controls"]]
    assert ids[0] == "female"
    assert "metal" in ids and "afrobeats" in ids
    assert len(set(ids)) == 16
    female = catalog["controls"][0]
    paths = control_paths(female)
    assert paths["native"].endswith("female_step1000.safetensors")
    assert paths["comfy"].endswith("female_step1000_comfyui.safetensors")
    card = release_card()
    assert card["seed"] == 1709
    assert card["scales"] == [0.0, 1.0]
    assert card["controls"][0]["comfy"] == paths["comfy"]


def test_hosts_are_music3_module_names():
    lm = resolve_host("particle")
    assert lm["name"] == "lm"
    assert target_replace(lm) == ["Qwen3Attention"]
    transformer = resolve_host("transformer")
    assert target_replace(transformer) == [
        "MiniMaxMusic3Attention",
        "MiniMaxMusic3TransformerBlock",
        "MiniMaxMusic3Transformer1DModel",
    ]
    assert target_replace(transformer, "attn") == ["MiniMaxMusic3Attention"]
    encoder = resolve_host("encoder")
    assert target_replace(encoder) == ["MiniMaxMusic3ConditionEncoder"]
