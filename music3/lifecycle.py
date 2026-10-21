"""Owned Music3 numerical response and native-only inference proposal.

The response registry is separate from images. Old0.6.0 CLI/checkpoint loaders
retain their original environment and source cohorts. No native model library
is imported and no tensor/model is constructed by registration or preflight.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re


MILESTONE = "21761ca8fd059439b58aa56a69bf8d1647f50e31"
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
    maximum = _integer(options, "max_sequence_length", 256, 1, 8192)
    if options.get("preset", "bispf_optimized") not in ("bispf", "bispf_optimized"):
        raise ValueError("unsupported PG preset")
    decay = options.get("ema_decay")
    if decay is not None and (type(decay) not in (int, float) or not math.isfinite(decay) or not 0 <= decay < 1):
        raise ValueError("EMA decay must be finite in [0,1)")
    _validate_model(options)
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


def _validate_model(options):
    """Shared train/inference gate; no teacher data or training construction."""
    dummy = options.get("dummy", False)
    if type(dummy) is not bool:
        raise ValueError("dummy must be boolean")
    width = _integer(options, "width", 32 if dummy else 4096, 1, 4096)
    _integer(options, "seed", 7, 0, 2**63 - 101)
    _integer(options, "max_sequence_length", 256, 1, 8192)
    device = options.get("device", "cpu")
    if not isinstance(device, str) or (device != "cpu" and not re.fullmatch(r"cuda:[0-9]+", device)):
        raise ValueError("device must be cpu or an explicit cuda index")
    if (options.get("pg_source_profile", "release"), options.get("penalty_input_domain", "auto")) != ("release", "auto"):
        raise UnsupportedOperation("this proposal uses official PG0.9.1 release/auto; recovered overlay qualification is pending")
    if width >= 170 or not dummy:
        raise UnsupportedOperation("native Music3 width4096 is gated: official PG0.9.1 has no explicit continuous selector; no projection or profile fallback")
    if device != "cpu" or options.get("base_precision", "float32") != "float32":
        raise UnsupportedOperation("Music3 random Qwen3 fixture requires CPU float32")
    if any(key in options for key in ("model_dir", "manifest", "manifest_sha256")):
        raise ValueError("random fixture cannot use native model identity")


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


def _build_native(options, *, checkpointing=False):
    """Build base and adapter only, independently of all training data/state."""
    _validate_model(options)
    import torch
    from particle_sliders.music3_native import ParticleNetwork
    from particle_sliders.pg09_audio import Music3ARResponseAdapter, require_audio_width

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
        if dummy:
            from particle_sliders.pg09 import _hash
            identity["frozen_base_sha256"] = _hash(model.state_dict())
        close = _restore_host(model)
        try:
            model.eval().requires_grad_(False)
            slider = ParticleNetwork(model)
            adapter = Music3ARResponseAdapter(model, slider, model_identity=identity,
                checkpoint_modules=tuple(model.model.layers) if checkpointing else (), **domain)
            return adapter, tokenizer, close
        except BaseException:
            close()
            raise


@contextmanager
def observe_native(model, network=None):
    """Preserve native buffers and modes beyond the adapter's common scope.

    Qwen3 forwards use_cache=False, so no KV history is produced. RoPE native
    caches and nonpersistent buffers may still change and belong to this hook.
    Frozen model weights are never cloned to observe a response.
    """
    import torch
    modules = tuple(model.modules())
    modes = [(module, module.training) for module in modules]
    buffers = [(module, dict(module._buffers), set(module._non_persistent_buffers_set)) for module in modules]
    values = {id(value): (value, value.detach().clone())
              for _, registered, _ in buffers for value in registered.values() if value is not None}
    attributes = [(module, {key: module.__dict__[key] for key in
                   ("max_seq_len_cached", "original_max_seq_len", "attention_scaling", "gradient_checkpointing")
                   if key in module.__dict__}) for module in modules]
    leaves = {id(parameter): parameter for root in (model, network) if root is not None for parameter in root.parameters()}
    parameters = [(parameter, parameter.requires_grad, parameter.grad,
                   None if parameter.grad is None else parameter.grad.detach().clone()) for parameter in leaves.values()]
    try:
        model.eval()
        yield
    finally:
        with torch.no_grad():
            for value, saved in values.values():
                if (value.shape, value.dtype, value.device) == (saved.shape, saved.dtype, saved.device):
                    value.copy_(saved)
                else:
                    value.data = saved.clone()
            for module, registered, nonpersistent in buffers:
                module._buffers.clear()
                module._buffers.update(registered)
                module._non_persistent_buffers_set.clear()
                module._non_persistent_buffers_set.update(nonpersistent)
            for module, previous in attributes:
                for key in ("max_seq_len_cached", "original_max_seq_len", "attention_scaling", "gradient_checkpointing"):
                    if key in previous:
                        setattr(module, key, previous[key])
                    else:
                        module.__dict__.pop(key, None)
            for parameter, flag, gradient, saved in parameters:
                parameter.requires_grad_(flag)
                if gradient is not None:
                    gradient.copy_(saved)
                parameter.grad = gradient
            for module, mode in modes:
                module.training = mode


class AudioProvider:
    """Owned frozen paired teachers with immutable identity and cloned fetches."""
    def __init__(self, adapter, rows):
        from particle_sliders.pg09_audio import audio_conditions
        self._rows = tuple(deepcopy(rows))
        self.response_shape = (1, int(adapter.model.config.hidden_size))
        self._conditions = audio_conditions(self._rows).detach().clone()

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError("Music3 provider index out of bounds")
        return deepcopy(self._rows[index])

    def conditions(self, indices):
        return self._conditions[indices.to(device=self._conditions.device)].detach().clone()

    def identity(self):
        return deepcopy(dict(protocol="music3-frozen-audio-provider-v1", rows=self._rows, conditions=self._conditions))


def _configuration(options):
    """Data-independent model specification; no teachers, budgets or paths."""
    dummy = options.get("dummy", False)
    return dict(dummy=dummy, width=options.get("width", 32 if dummy else 4096),
        seed=options.get("seed", 7), max_sequence_length=options.get("max_sequence_length", 256),
        base_precision=options.get("base_precision", "float32" if dummy else "bfloat16"),
        pg_source_profile=options.get("pg_source_profile", "release"),
        penalty_input_domain=options.get("penalty_input_domain", "auto"))


def prepare(options):
    """Owned provider/game Binding through the shared response lifecycle."""
    validate(options)
    import torch
    from particle_sliders import ConditionalPG09
    from particle_sliders.lifecycle import Binding
    from particle_sliders.pg09_audio import AudioResponseCritic, AudioRow
    from particle_sliders.music3_native import assemble_text, tokenize_text, assert_audio_start
    adapter, tokenizer, close = _build_native(options, checkpointing=options.get("checkpointing", False))
    try:
        rows = []
        with torch.random.fork_rng(devices=[]), observe_native(adapter.model, adapter.network):
            torch.set_rng_state(torch.Generator().manual_seed(options.get("seed", 7) + 1).get_state())
            for row in options.get("rows", deepcopy(_FIXTURE_ROWS)):
                if options.get("dummy"):
                    neutral, positive = tuple(row["neutral_ids"]), tuple(row["positive_ids"])
                    history = row.get("history_tokens", 0)
                else:  # Retain native packing for a future separately qualified pin.
                    history_ids = tuple(row.get("history_ids", []))
                    values = []
                    for key in ("neutral", "positive"):
                        ids, mask = tokenize_text(tokenizer, assemble_text(row[key], row.get("lyrics", "")), adapter.model.device)
                        assert_audio_start(ids, mask, tokenizer, where="Music3 owned factory")
                        values.append(tuple(ids[0][mask[0].bool()].tolist()) + history_ids)
                    neutral, positive = values
                    history = len(history_ids)
                if (max(len(neutral), len(positive)) > options.get("max_sequence_length", 256)
                        or max(neutral + positive) >= adapter.model.config.vocab_size):
                    raise ValueError("complete native context exceeds token budget; no truncation")
                rows.append(adapter.prepare(AudioRow(neutral, positive, history_tokens=history)))
            provider = AudioProvider(adapter, rows)
            # Existing definition retained until matched-weight HNDL mapping/parity.
            critic = AudioResponseCritic(provider.response_shape[1], provider._conditions.shape[1]).to(adapter.model.device)
            game = ConditionalPG09(adapter, provider, critic,
                particles=adapter.particles, response_domain="continuous_vector",
                preset=options.get("preset", "bispf_optimized"), batch_size=options.get("batch", 2),
                total_steps=options.get("steps", 4), seed=options.get("seed", 7),
                pg_source_profile=options.get("pg_source_profile", "release"),
                penalty_input_domain=options.get("penalty_input_domain", "auto"))
            specification = INFERENCE.specification(adapter, model_identity=adapter.model_identity,
                protocol="music3-native-final-ar-hidden-v1", configuration=_configuration(options))
        return Binding(game, close, inference=specification, observation=lambda: observe_native(adapter.model, adapter.network))
    except BaseException:
        close()
        raise


def capabilities(options):
    dummy = options.get("dummy", False)
    width = options.get("width", 32 if dummy else 4096)
    operations = ("train", "resume", "evaluate", "export", "response")
    supported = (dummy is True and type(width) is int and 1 <= width < 170
                 and options.get("pg_source_profile", "release") == "release"
                 and options.get("penalty_input_domain", "auto") == "auto")
    unavailable = (("render", "Music3 acoustic/token sampling is unqualified; numerical hidden responses only"),)
    if not supported:
        reason = "Music3 native/wide AR responses are gated pending explicit continuous-domain source qualification"
        unavailable += tuple((operation, reason) for operation in operations)
    return ResponseCapabilities("routed_particle_gan", "audio_ar_hidden", "native_final_ar_hidden",
        "continuous_vector", (1, width), "0.9.1",
        options.get("pg_source_profile", "release"), options.get("penalty_input_domain", "auto"),
        base_precisions=("float32",) if dummy else ("bfloat16",),
        operations=operations if supported else (), unavailable_reasons=unavailable,
        qualification="Music3 source proposal; tiny response conformance pending, native width and acoustic quality gated")


def preflight(options, request, staged):
    if staged or request.get("native") or request.get("output_type") is not None:
        raise UnsupportedOperation("Music3 owns its factory; training accepts no staged objects or media requests")


def _inference_options(options, spec):
    if set(options) - {"device", "model_dir", "manifest", "manifest_sha256"}:
        raise ValueError("inference runtime accepts only local model locations/device; no training data or configuration overrides")
    config = spec.configuration
    if set(config) != {"dummy", "width", "seed", "max_sequence_length", "base_precision", "pg_source_profile", "penalty_input_domain"}:
        raise ValueError("unsupported Music3 native inference configuration")
    return dict(config, **options)


def validate_inference(options, spec):
    merged = _inference_options(options, spec)
    _validate_model(merged)
    from particle_sliders.pg09_audio import require_audio_width
    require_audio_width(merged["width"], pg_source_profile=merged["pg_source_profile"],
                        penalty_input_domain=merged["penalty_input_domain"])


def _probe_ids(options, native):
    if set(native) != {"ids"}:
        raise ValueError("Music3 fixture inference requires an unpadded ids array; no image/waveform inputs")
    ids = _ids(native["ids"])
    if max(ids) >= 256 or len(ids) > options["max_sequence_length"]:
        raise ValueError("complete response request exceeds native token budget; no truncation")
    return ids


def preflight_inference(options, request, spec):
    if request.get("output_type") is not None:
        raise UnsupportedOperation("Music3 numerical responses have no media output type")
    native = request.get("native", {})
    # Opening without a request is allowed; response() must supply fresh IDs.
    if native:
        _probe_ids(_inference_options(options, spec), native)


def prepare_inference(options, spec):
    """Base/adapter only: never construct provider, teachers, critic or PG game."""
    validate_inference(options, spec)
    from particle_sliders.inference import InferenceBinding, native_identity
    merged = _inference_options(options, spec)
    adapter, _, close = _build_native(merged)
    try:
        identity = native_identity(adapter, model_identity=adapter.model_identity,
                                   protocol="music3-native-final-ar-hidden-v1")
        def response(*, strength=1., seed=0, **native):
            del seed  # Common inference scope owns deterministic observation RNG.
            return adapter._response(_probe_ids(merged, native), scale=strength).detach().clone()
        return InferenceBinding(adapter.network, identity, close, response=response,
                                observation=lambda: observe_native(adapter.model, adapter.network))
    except BaseException:
        close()
        raise


# Registry construction imports the reviewed shared contract, never transformers.
from .response_source import verify_response_core
verify_response_core()
from particle_sliders.lifecycle import UnsupportedOperation
from particle_sliders.response import ResponseBackend, ResponseCapabilities
from particle_sliders.inference import InferenceBackend
verify_response_core()  # Also verify the origins of the just-imported modules.

BACKEND = ResponseBackend("music3", capabilities({}), prepare, validate,
                          configure_capabilities=capabilities, preflight=preflight)
INFERENCE = InferenceBackend("music3", capabilities({}), prepare_inference, validate_inference,
                            configure_capabilities=capabilities, preflight=preflight_inference)
