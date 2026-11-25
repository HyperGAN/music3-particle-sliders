from fractions import Fraction
import os
from zipfile import ZipFile

import pytest

from music3.semantic_dose import SemanticDoseMatcher


@pytest.fixture
def dose_wheel():
    wheel = os.environ.get("MUSIC3_CORE_338_WHEEL")
    if not wheel:
        pytest.skip("Provide MUSIC3_CORE_338_WHEEL for the optional verified interval-dose module")
    return wheel


@pytest.fixture
def matcher(dose_wheel):
    return SemanticDoseMatcher(dose_wheel)


def test_actual_constant_can_match_on_a_different_returned_clock(matcher):
    gate = (0, 0, 1, 0, 0)
    proposal = matcher.propose(gate, expected_eligible=2)
    readback = matcher.readback(gate, (0, proposal, proposal))

    assert proposal == 0.5
    assert readback.status == "exact"
    assert (readback.adaptive_eligible, readback.constant_eligible) == (4, 2)
    assert readback.adaptive_dose == readback.constant_dose == Fraction(1)
    assert readback.dose_error == 0


def test_nearest_control_reports_the_exact_binary32_residual(matcher):
    gate = (0,) + (1,) * 3 + (0,) * 7
    proposal = matcher.propose(gate, expected_eligible=10)
    readback = matcher.readback(gate, (0,) + (0.3,) * 10)

    assert proposal.hex() == "0x1.3333340000000p-2"
    assert readback.status == "nearest"
    assert readback.nearest_strength == proposal
    assert readback.adaptive_dose == Fraction(3)
    assert readback.dose_error == Fraction(1, 8388608)


def test_premature_eos_uses_the_returned_cached_predictions(matcher):
    gate = (0, 1, 0, 0, 0)
    proposal = matcher.propose(gate, expected_eligible=4)
    readback = matcher.readback(gate, (0, proposal, proposal, proposal))

    assert proposal == 0.25
    assert readback.status == "unmatched"
    assert readback.constant_eligible == 3
    assert readback.dose_error == Fraction(-1, 4)
    assert readback.nearest_strength == matcher.propose(gate, expected_eligible=3)


def test_short_return_may_have_no_feasible_constant(matcher):
    gate = (0, 1, 1, 1)
    assert matcher.propose(gate, expected_eligible=2) is None
    readback = matcher.readback(gate, (0, 1, 1))
    assert readback.status == "unmatched"
    assert readback.nearest_strength is None
    assert readback.dose_error == -1


def test_zero_dose_matches_even_when_eos_lengths_differ(matcher):
    gate = (0, 0, 0)
    assert matcher.propose(gate, expected_eligible=1) == 0
    assert matcher.readback(gate, (0, 0)).status == "exact"


@pytest.mark.parametrize("applied", [(), (0,), (1, 0), (0, float("nan")), (0, -0.1), (0, 1.1), (0, True)])
def test_refuses_malformed_applied_strengths(matcher, applied):
    with pytest.raises(ValueError):
        matcher.propose(applied, expected_eligible=2)
    with pytest.raises(ValueError):
        matcher.readback((0, 1), applied)


def test_refuses_nonconstant_receipt_and_invalid_hypothesized_count(matcher):
    with pytest.raises(ValueError, match="single resolved cached strength"):
        matcher.readback((0, 1), (0, 0.25, 0.5))
    with pytest.raises(ValueError, match="positive integer"):
        matcher.propose((0, 1), expected_eligible=0)


def test_refuses_altered_wheel_source_even_after_loading_the_verified_wheel(dose_wheel, tmp_path):
    SemanticDoseMatcher(dose_wheel)
    member = "particle_sliders/strength_schedule.py"
    with ZipFile(dose_wheel) as archive:
        source = archive.read(member)
    changed = tmp_path / "altered.whl"
    with ZipFile(changed, "w") as archive:
        archive.writestr(member, b" " + source[1:])

    with pytest.raises(ValueError, match="exact shared interval-dose source"):
        SemanticDoseMatcher(changed)
