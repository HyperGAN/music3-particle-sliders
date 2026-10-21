"""Opt-in Music3 factory draft for the reviewed shared lifecycle milestone.

This module deliberately has no image backend registration. The shared owner
must first provide an honest AR-hidden-response lifecycle contract. Existing
0.6.0 CLI/checkpoint loaders remain the authority for their source cohorts.
Import and describe perform no tensor allocation or native dependency imports.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re


MILESTONE = "fd89e1c780ab5210cd1be98ec3b70977e6349379"
MODEL_REVISION = "bd348f9c49ea3c1b39f33ace3436f8fad435f24e"
_OPTIONS = {
    "dummy", "host", "rows", "steps", "batch", "seed", "width", "device",
    "base_precision", "ema_decay", "preset", "checkpointing", "max_sequence_length",
    "pg_source_profile", "penalty_input_domain", "model_dir", "manifest", "manifest_sha256",
}
_FIXTURE_ROWS = (
    {"neutral_ids": [11, 13, 17, 19], "positive_ids": [29, 31, 17, 19], "history_tokens": 2},
    {"neutral_ids": [23, 37, 41, 43, 47], "positive_ids": [53, 59, 41, 43, 47], "history_tokens": 3},
)


def _integer(options, name, default, low, high):
    value = options.get(name, default)
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    return value


def _ids(value, *, empty=False):
    if (not isinstance(value, (list, tuple)) or (not value and not empty)
            or any(type(token) is not int or token < 0 for token in value)):
        raise ValueError("token IDs must be nonnegative integers")
    return tuple(value)


def validate(options):
    """Check configuration and small local metadata before model allocation."""
    if not isinstance(options, dict) or set(options) - _OPTIONS:
        raise ValueError("unknown Music3 factory options; legacy loader options are unsupported")
    for name, default in (("dummy", False), ("checkpointing", False)):
        if type(options.get(name, default)) is not bool:
            raise ValueError(name + " must be boolean")
    dummy = options.get("dummy", False)
    if options.get("host", "lm") not in ("lm", "particle"):
        raise ValueError("only the Music3 LM particle host is bound; transformer/encoder are unsupported")
    _integer(options, "steps", 4, 1, 1000000)
    _integer(options, "batch", 2, 2, 512)
    _integer(options, "seed", 7, 0, 2**63 - 101)
    width = _integer(options, "width", 32 if dummy else 4096, 1, 4096)
    maximum = _integer(options, "max_sequence_length", 256, 1, 8192)
    device = options.get("device", "cpu")
    if not isinstance(device, str) or (device != "cpu" and not re.fullmatch(r"cuda:[0-9]+", device)):
        raise ValueError("device must be cpu or an explicit cuda index in an admitted environment")
    if options.get("preset", "bispf_optimized") not in ("bispf", "bispf_optimized"):
        raise ValueError("unsupported PG preset")
    decay = options.get("ema_decay")
    if decay is not None and (type(decay) not in (int, float) or not math.isfinite(decay) or not 0 <= decay < 1):
        raise ValueError("EMA decay must be finite in [0,1)")
    profile = (options.get("pg_source_profile", "release"), options.get("penalty_input_domain", "auto"))
    if profile not in (("release", "auto"), ("alz-continuous-v1", "continuous")):
        raise ValueError("source profile and penalty domain must be an explicit reviewed pair")
    if width >= 170 and profile != ("alz-continuous-v1", "continuous"):
        raise ValueError("native wide AR vectors require the reviewed continuous profile; no projection fallback")
    precision = options.get("base_precision", "float32" if dummy else "bfloat16")
    if dummy:
        if options.get("device", "cpu") != "cpu" or precision != "float32":
            raise ValueError("Music3 random Qwen3 fixture requires CPU float32")
        if any(key in options for key in ("model_dir", "manifest", "manifest_sha256")):
            raise ValueError("random fixture cannot use native model identity")
    else:
        if precision != "bfloat16" or width != 4096:
            raise ValueError("native Music3 is BF16 with its full 4096-wide AR state")
        _native_record(options)
    rows = options.get("rows", deepcopy(_FIXTURE_ROWS) if dummy else None)
    if not isinstance(rows, (list, tuple)) or not 2 <= len(rows) <= 128:
        raise ValueError("Music3 requires 2..128 paired rows")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Music3 row must be an object")
        if dummy:
            if set(row) - {"neutral_ids", "positive_ids", "history_tokens"}:
                raise ValueError("random fixture rows use explicit synthetic token IDs")
            neutral, positive = _ids(row.get("neutral_ids")), _ids(row.get("positive_ids"))
            history = row.get("history_tokens", 0)
            if (type(history) is not int or not 0 <= history < min(len(neutral), len(positive))
                    or neutral == positive or (history and neutral[-history:] != positive[-history:])):
                raise ValueError("distinct prefixes must share the declared continuation history")
            if max(neutral + positive) >= 256 or max(len(neutral), len(positive)) > maximum:
                raise ValueError("complete fixture context exceeds its native token budget")
        else:
            if set(row) - {"neutral", "positive", "lyrics", "history_ids"}:
                raise ValueError("native rows require paired caption/lyrics and optional shared history IDs")
            if any(not isinstance(row.get(key), str) or not row[key].strip() for key in ("neutral", "positive")):
                raise ValueError("nonempty neutral/positive captions required")
            if row["neutral"] == row["positive"] or not isinstance(row.get("lyrics", ""), str):
                raise ValueError("distinct captions and string lyrics required")
            _ids(row.get("history_ids", []), empty=True)


def describe(options, *, operation="train", strength=1., references=False):
    """Allocation-free modality contract; this never labels hidden states audio."""
    # Unsupported requests fail even before reading local configuration files.
    if operation not in ("train", "resume", "evaluate", "response", "export"):
        raise ValueError("Music3 acoustic/token rendering is unqualified; only hidden-response operations are supported")
    if references or type(strength) not in (int, float) or not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Music3 hidden responses support no image references and finite strengths in [0,1]")
    validate(options)
    return dict(
        backend="music3", modality="audio_ar_hidden", response_domain="continuous_vector",
        response_shape=[1, options.get("width", 32 if options.get("dummy") else 4096)],
        generator_arm="routed_particle_gan", particles=[128, 4], adapter_precision="float32",
        trained_scales=[1.], response_strengths=[0., 1.], response_output="native_final_ar_hidden",
        time_coordinates="AR continuation history length; no diffusion scheduler",
        student_conditioning="neutral caption/lyrics with shared continuation IDs",
        target="frozen positive final AR state", native_render=False,
        qualification="source draft; conformance and native sampling qualification pending",
        shared_milestone=MILESTONE,
    )


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _native_record(options):
    """Read the existing staged manifest without inheriting old cohort identity."""
    for key in ("model_dir", "manifest", "manifest_sha256"):
        if not isinstance(options.get(key), str) or not options[key]:
            raise ValueError("native factory requires local " + key)
    if not re.fullmatch(r"[0-9a-f]{64}", options["manifest_sha256"]):
        raise ValueError("manifest_sha256 must be the reviewed digest")
    manifest = Path(options["manifest"])
    if manifest.stat().st_size > 1024 * 1024 or _sha_file(manifest) != options["manifest_sha256"]:
        raise ValueError("native staged manifest differs")
    record = json.loads(manifest.read_text())
    if record.get("revision") != MODEL_REVISION or not isinstance(record.get("files"), list):
        raise ValueError("native staged manifest must bind the reviewed Music3 revision and files")
    files = {}
    for entry in record["files"]:
        name = entry.get("filename", "")
        path = PurePosixPath(name)
        if (not name or path.is_absolute() or ".." in path.parts or "\\" in name
                or path.as_posix() != name):
            raise ValueError("manifest requires canonical relative file names")
        if path.parts[0] not in ("language_model", "tokenizer"):
            continue  # Acoustic weights are outside this LM-only factory.
        digest = entry.get("sha256", "")
        if name in files or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("duplicate or invalid native file digest")
        files[name] = digest
    root = Path(options["model_dir"])
    actual = {str(p.relative_to(root)) for folder in ("language_model", "tokenizer")
              for p in (root / folder).rglob("*") if p.is_file()}
    if (not files or actual != set(files) or "language_model/config.json" not in files
            or "tokenizer/tokenizer_config.json" not in files
            or not any(name.endswith(".safetensors") for name in files)):
        raise ValueError("native manifest must exactly cover all staged LM/tokenizer files")
    config_path = root / "language_model/config.json"
    if _sha_file(config_path) != files["language_model/config.json"]:
        raise ValueError("native Qwen3 config digest differs")
    config = json.loads(config_path.read_text())
    if config.get("model_type") != "qwen3" or config.get("hidden_size") != 4096:
        raise ValueError("native Music3 requires its genuine 4096-wide Qwen3 model")
    return files


def _restore_host(model):
    """Unwind exact forward attributes, checkpoint wrappers, flags and modes."""
    keys = ("forward", "gradient_checkpointing", "_gradient_checkpointing_func", "_pg09_scale_checkpoint")
    saved = [(module, module.training, {key: module.__dict__[key] for key in keys if key in module.__dict__})
             for module in model.modules()]
    flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    closed = False

    def close():
        nonlocal closed
        if closed:
            return
        closed = True
        for module, mode, previous in saved:
            for key in keys:
                if key in previous:
                    setattr(module, key, previous[key])
                else:
                    module.__dict__.pop(key, None)
            module.training = mode
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
    return close


def prepare(options):
    """Own loading, real routed adapter, frozen teachers, critic/game and cleanup.

    Returns only the shared Binding. No product session, GAN loop, optimizer,
    prior, sampler or checkpoint implementation is introduced here. Native
    rendering remains absent until its product sampling callback is qualified.
    """
    describe(options)
    import torch
    from particle_sliders import ConditionalPG09
    from particle_sliders.lifecycle import Binding
    from particle_sliders.music3_native import ParticleNetwork
    from particle_sliders.pg09_audio import AudioResponseCritic, AudioRow, Music3ARResponseAdapter, audio_conditions, require_audio_width

    dummy = options.get("dummy", False)
    width = options.get("width", 32 if dummy else 4096)
    domain = dict(pg_source_profile=options.get("pg_source_profile", "release"),
                  penalty_input_domain=options.get("penalty_input_domain", "auto"))
    require_audio_width(width, **domain)  # Exact PG source gate before model allocation.
    device = torch.device(options.get("device", "cpu"))
    devices = [device.index or 0] if device.type == "cuda" else []
    # Construction never consumes the caller's CPU or selected-device RNG.
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(torch.Generator().manual_seed(options.get("seed", 7)).get_state())
        if devices:
            torch.cuda.set_rng_state(torch.Generator(device=device).manual_seed(options.get("seed", 7)).get_state(), device)
        tokenizer = None
        if dummy:
            from transformers import Qwen3Config, Qwen3ForCausalLM
            model = Qwen3ForCausalLM(Qwen3Config(hidden_size=width, intermediate_size=64,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                vocab_size=256, max_position_embeddings=options.get("max_sequence_length", 256),
                attention_dropout=0., use_cache=False, attn_implementation="eager"))
            identity = dict(fixture="random Qwen3; synthetic token IDs", seed=options.get("seed", 7), width=width)
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            root = Path(options["model_dir"])
            files = _native_record(options)
            for name, digest in files.items():
                if _sha_file(root / name) != digest:
                    raise ValueError("native staged file digest differs: " + name)
            tokenizer = AutoTokenizer.from_pretrained(str(root / "tokenizer"), local_files_only=True, trust_remote_code=False)
            # The shared helper permits an unknown token when a tokenizer cannot
            # resolve the marker. This fixed native revision requires its real
            # audio-start marker before any teacher context is assembled.
            if tokenizer.convert_tokens_to_ids("<|audio_start|>") != 151669:
                raise ValueError("native Music3 tokenizer must resolve audio-start token 151669")
            model = AutoModelForCausalLM.from_pretrained(str(root / "language_model"),
                local_files_only=True, trust_remote_code=False, use_safetensors=True,
                torch_dtype=torch.bfloat16, attn_implementation="eager")
            identity = dict(manifest_sha256=options["manifest_sha256"], revision=MODEL_REVISION, files=files)
        model = model.to(device)
        close = _restore_host(model)
        try:
            model.eval().requires_grad_(False)
            slider = ParticleNetwork(model)
            adapter = Music3ARResponseAdapter(model, slider, model_identity=identity,
                checkpoint_modules=tuple(model.model.layers) if options.get("checkpointing", False) else (), **domain)
            rows = []
            maximum = options.get("max_sequence_length", 256)
            from particle_sliders.music3_native import assemble_text, tokenize_text, assert_audio_start
            for row in options.get("rows", deepcopy(_FIXTURE_ROWS)):
                if dummy:
                    neutral, positive = tuple(row["neutral_ids"]), tuple(row["positive_ids"])
                    history = row.get("history_tokens", 0)
                else:
                    history_ids = tuple(row.get("history_ids", []))
                    values = []
                    for key in ("neutral", "positive"):
                        ids, mask = tokenize_text(tokenizer, assemble_text(row[key], row.get("lyrics", "")), device)
                        assert_audio_start(ids, mask, tokenizer, where="Music3 owned factory")
                        values.append(tuple(ids[0][mask[0].bool()].tolist()) + history_ids)
                    neutral, positive = values
                    history = len(history_ids)
                if (max(len(neutral), len(positive)) > maximum
                        or max(neutral + positive) >= model.config.vocab_size):
                    raise ValueError("complete native context exceeds token budget; no truncation")
                rows.append(adapter.prepare(AudioRow(neutral, positive, history_tokens=history)))
            conditions = audio_conditions(rows)
            # Keep the old matched network definition until common HNDL migration
            # proves state mapping and update parity. No arbitrary replacement.
            critic = AudioResponseCritic(width, conditions.shape[1]).to(device)
            game = ConditionalPG09(adapter, rows, critic, conditions=conditions,
                particles=adapter.particles, response_domain="continuous_vector",
                preset=options.get("preset", "bispf_optimized"), batch_size=options.get("batch", 2),
                total_steps=options.get("steps", 4), seed=options.get("seed", 7), **domain)
            return Binding(game, close, render=None)
        except BaseException:
            close()
            raise
