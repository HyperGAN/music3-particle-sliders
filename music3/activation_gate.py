"""Shared activation steering for Music3's current and captured RAW owners."""

from dataclasses import asdict
from hashlib import sha256
import sys
from threading import Lock
from types import ModuleType
from zipfile import ZipFile

import torch


CORE_REVISION = "b1f780decc516808cfd3019d5e5b674457161f86"
CORE_SOURCE = "particle_sliders/adaptive_steering.py"
CORE_SOURCE_SHA256 = "5fc2a5256a935eca9cecc365d9ac62e5e94cdc375d2ab33c5dc26eb344241814"
_MODULE_NAME = f"{__name__}._core_{CORE_REVISION}"
_LOAD_LOCK = Lock()


def load_steering_core(core_wheel):
    """Load the exact standalone shared equations without replacing the RAW owner's core."""
    with ZipFile(core_wheel) as archive:
        entry = archive.getinfo(CORE_SOURCE)
        if entry.file_size != 10329:
            raise ValueError("Music3 requires the exact shared steering module from core PR #269")
        source = archive.read(entry)
    if sha256(source).hexdigest() != CORE_SOURCE_SHA256:
        raise ValueError("Music3 requires the exact shared steering module from core PR #269")
    with _LOAD_LOCK:
        if _MODULE_NAME not in sys.modules:
            module = ModuleType(_MODULE_NAME)
            module.__file__ = f"{core_wheel}/{CORE_SOURCE}"
            sys.modules[_MODULE_NAME] = module
            try:
                exec(compile(source, module.__file__, "exec"), module.__dict__)
            except BaseException:
                del sys.modules[_MODULE_NAME]
                raise
        return sys.modules[_MODULE_NAME]


class SemanticGatePlan:
    """Calibration and probe for the live frame-feedback input to the Music3 LM."""

    def __init__(self, *, core_wheel, method, parameter, calibration_sha256,
                 direction=None, seed=None, clabs=None):
        if (not isinstance(calibration_sha256, str) or len(calibration_sha256) != 64
                or any(ch not in "0123456789abcdef" for ch in calibration_sha256)):
            raise ValueError("Provide the SHA-256 of the Music3 gate calibration record")
        self._core = load_steering_core(core_wheel)
        self._clabs = self._core.CLABSParameters(**clabs) if clabs is not None else None
        self._method, self._parameter, self._seed = method, float(parameter), seed
        self._calibration_sha256 = calibration_sha256
        if method == "random":
            if direction is not None:
                raise ValueError("The matched random gate does not use a direction")
            self._direction = None
        else:
            if direction is None:
                raise ValueError("Music3 CFP and CLABS require a calibrated frame-feedback direction")
            probe = torch.as_tensor(direction, dtype=torch.float32).detach().clone()
            if (probe.ndim != 1 or not torch.isfinite(probe).all()
                    or not torch.isclose(probe.norm(), probe.new_tensor(1.0))):
                raise ValueError("Music3 requires a finite unit direction in frame-feedback space")
            self._direction = probe
        self._new_gate()

    def _new_gate(self):
        return self._core.ActivationGate(self._method, self._parameter, seed=self._seed,
                                         clabs=self._clabs)

    def _project(self, embeds, strength):
        if self._direction is None:
            return None
        if self._direction.shape != (embeds.shape[-1],):
            raise ValueError("Direction width must match the native Music3 frame embedding")
        return float(self._core.activation_projection(embeds[0, 0], self._direction,
                                                       strength=strength))

    def _receipt(self):
        return dict(
            core_revision=CORE_REVISION,
            signal="conditional row of the native pre-edit cached frame-feedback embedding",
            calibration_sha256=self._calibration_sha256,
            method=self._method, parameter=self._parameter, seed=self._seed,
            direction=self._direction.tolist() if self._direction is not None else None,
            clabs=asdict(self._clabs) if self._clabs is not None else None,
        )
