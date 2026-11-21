"""Per-evaluation RAW strength on the resident Music3 semantic pipeline."""

from contextlib import ExitStack
from math import isfinite
from sys import float_info


def _max_frames(pipeline, audio_duration):
    from diffusers.modular_pipelines.minimax_music3.encoders import _MAX_AUDIO_FRAMES

    if type(audio_duration) not in (int, float) or not 0 < audio_duration <= float_info.max:
        raise ValueError("audio duration must be positive and finite")
    return int(min(audio_duration * pipeline.frame_rate, _MAX_AUDIO_FRAMES))


def generate(pipeline, *, scale_adapter, audio_duration, step_gains, strength=1.0, **inputs):
    """Return the original semantic stage result and the gains used before EOS."""
    max_frames = _max_frames(pipeline, audio_duration)
    if (max_frames < 1 or not isinstance(step_gains, (list, tuple))
            or len(step_gains) != max_frames + 1):
        raise ValueError("provide one prefill gain and one gain for each possible frame")
    if type(strength) not in (int, float) or not 0 <= strength <= 1:
        raise ValueError("nominal Music3 strength must be finite in [0,1]")
    if any(type(gain) not in (int, float) or not 0 <= gain <= float_info.max for gain in step_gains):
        raise ValueError("Music3 step gains must be finite and nonnegative")
    strengths = [float(strength * gain) for gain in step_gains]
    if any(not isfinite(value) or value > 1 for value in strengths):
        raise ValueError("effective Music3 strengths must be finite in [0,1]")

    def resolve(position, _embeds):
        return strengths[position]

    return _run_semantic(pipeline, scale_adapter, audio_duration, max_frames, resolve, inputs)


def generate_gated(pipeline, *, scale_adapter, audio_duration, gate_plan, strength, **inputs):
    """Return native semantic output and one trace entry per cached frame prediction."""
    from music3.activation_gate import SemanticGatePlan

    max_frames = _max_frames(pipeline, audio_duration)
    if max_frames < 1:
        raise ValueError("audio duration must cover at least one possible Music3 frame")
    if type(strength) not in (int, float) or not 0 < strength <= 1:
        raise ValueError("gated Music3 requires an explicit nonzero nominal strength in (0,1]")
    if not isinstance(gate_plan, SemanticGatePlan):
        raise TypeError("gated Music3 requires one calibrated SemanticGatePlan")

    gate = gate_plan._new_gate()
    decisions = []

    def resolve(position, embeds):
        # Prefill only predicts the warm-up code. It emits no output frame.
        if position == 0:
            return 0.0
        projection = gate_plan._project(embeds, strength)
        decision = gate.step(projection)
        effective = float(strength) if decision.fired else 0.0
        decisions.append(dict(frame_prediction=position - 1, projection=projection,
                              probability=decision.probability, fired=decision.fired,
                              effective_strength=effective))
        return effective

    result, applied = _run_semantic(pipeline, scale_adapter, audio_duration, max_frames, resolve, inputs)
    usage = gate.usage
    receipt = dict(activation_gate=gate_plan._receipt(), decisions=decisions,
                   applied_strengths=applied, nominal_strength=float(strength),
                   fired=usage.fired, eligible=usage.eligible,
                   gate_rate=usage.rate if usage.eligible else None,
                   total_strength=sum(applied))
    return result, receipt


def _run_semantic(pipeline, scale_adapter, audio_duration, max_frames, resolve, inputs):
    text_model = pipeline.language_model.model
    scope = ExitStack()
    applied = []
    prefill_length = None

    def before(_module, _args, request):
        nonlocal prefill_length
        position = len(applied)
        embeds = request.get("inputs_embeds")
        cache = request.get("past_key_values")
        length = 0 if cache is None else int(cache.get_seq_length())
        if (embeds is None or embeds.ndim != 3 or embeds.shape[0] != 2
                or not request.get("use_cache") or position > max_frames):
            raise RuntimeError("Music3 requires its native batched CFG semantic evaluations")
        if position == 0:
            if length != 0 or embeds.shape[1] == 0:
                raise RuntimeError("Music3 semantic prefill must start with an empty cache")
            prefill_length = embeds.shape[1]
        elif embeds.shape[1] != 1 or length != prefill_length + position - 1:
            raise RuntimeError("Music3 cached frame position differs from the requested schedule")
        value = resolve(position, embeds)
        scope.enter_context(scale_adapter(value))
        applied.append(value)

    def after(_module, _args, _output):
        scope.close()

    with ExitStack() as request:
        request.callback(text_model.register_forward_pre_hook(before, with_kwargs=True).remove)
        request.callback(text_model.register_forward_hook(after, always_call=True).remove)
        request.callback(scope.close)
        result = pipeline(audio_duration=audio_duration, **inputs)
    if not applied:
        raise RuntimeError("Music3 semantic generation used no scheduled evaluations")
    return result, applied
