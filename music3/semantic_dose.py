"""Read Music3 coefficient dose on the cached semantic prediction clock."""

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from struct import Struct
import sys
from threading import Lock
from types import ModuleType
from typing import Literal
from zipfile import ZipFile


_CORE_REVISION = "34ed1856f224f9a32ab092e58d9803f0cfc45c38"
_CORE_SOURCE = "particle_sliders/strength_schedule.py"
_CORE_SOURCE_SHA256 = "fed34a2d3f70fd793f8b13cb605e80442699912edb3cab3258c1e21f92e50282"
_CORE_MODULE = f"{__name__}._core_{_CORE_REVISION}"
_LOAD_LOCK = Lock()
_FLOAT32 = Struct(">f")


def _load_dose_core(core_wheel):
    with ZipFile(core_wheel) as archive:
        entry = archive.getinfo(_CORE_SOURCE)
        if entry.file_size != 7125:
            raise ValueError("Music3 requires the exact shared interval-dose source from core PR #338")
        source = archive.read(entry)
    if sha256(source).hexdigest() != _CORE_SOURCE_SHA256:
        raise ValueError("Music3 requires the exact shared interval-dose source from core PR #338")
    with _LOAD_LOCK:
        if _CORE_MODULE not in sys.modules:
            module = ModuleType(_CORE_MODULE)
            module.__file__ = f"{core_wheel}/{_CORE_SOURCE}"
            sys.modules[_CORE_MODULE] = module
            try:
                exec(compile(source, module.__file__, "exec"), module.__dict__)
            except BaseException:
                del sys.modules[_CORE_MODULE]
                raise
        return sys.modules[_CORE_MODULE]


def _cached_coefficients(applied_strengths):
    applied = tuple(applied_strengths)
    if any(type(strength) not in (int, float) or not 0 <= strength <= 1 for strength in applied):
        raise ValueError("Music3 applied strengths must be finite numbers in [0,1]")
    if len(applied) < 2 or applied[0] != 0:
        raise ValueError("provide zero prefill and at least one returned cached prediction")
    return tuple(_FLOAT32.unpack(_FLOAT32.pack(strength))[0] for strength in applied[1:])


@dataclass(frozen=True)
class SemanticDoseReadback:
    """Comparison of actual returned strengths in binary32 coefficient space."""

    status: Literal["exact", "nearest", "unmatched"]
    adaptive_eligible: int
    constant_eligible: int
    adaptive_dose: Fraction
    constant_dose: Fraction
    nearest_strength: float | None

    @property
    def dose_error(self):
        return self.constant_dose - self.adaptive_dose


class SemanticDoseMatcher:
    """Propose a constant and verify receipts with a separate, pinned core wheel."""

    def __init__(self, core_wheel):
        self._core = _load_dose_core(core_wheel)

    def _nearest(self, coefficients, eligible):
        match = self._core.match_binary32_interval_dose(
            coefficients, range(len(coefficients) + 1), constant_coordinates=(0, eligible),
        )
        # A shorter native run may make the target unreachable within [0,1].
        if match.adaptive_dose > eligible:
            return None
        return match.constant_strength

    def propose(self, adaptive_applied_strengths, expected_eligible):
        """Return the nearest constant for a hypothesized cached length, if feasible."""
        if type(expected_eligible) is not int or expected_eligible < 1:
            raise ValueError("expected eligible predictions must be a positive integer")
        return self._nearest(_cached_coefficients(adaptive_applied_strengths), expected_eligible)

    def readback(self, adaptive_applied_strengths, constant_applied_strengths):
        """Measure both returned clocks. The nearest label describes dose only."""
        adaptive = _cached_coefficients(adaptive_applied_strengths)
        constant = _cached_coefficients(constant_applied_strengths)
        if any(strength != constant[0] for strength in constant):
            raise ValueError("the returned control must have a single resolved cached strength")

        adaptive_dose = self._core.interval_dose(
            adaptive, range(len(adaptive) + 1), coefficient_format="binary32",
        )
        constant_dose = self._core.interval_dose(
            constant, range(len(constant) + 1), coefficient_format="binary32",
        )
        nearest = self._nearest(adaptive, len(constant))
        if constant_dose == adaptive_dose:
            status = "exact"
        elif constant[0] == nearest:
            status = "nearest"
        else:
            status = "unmatched"
        return SemanticDoseReadback(
            status, len(adaptive), len(constant), adaptive_dose, constant_dose, nearest,
        )
