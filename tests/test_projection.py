import math

from finplan.tools.projection import project_growth


def test_zero_return_is_just_contributions():
    r = project_growth(initial=1000, monthly=100, years=1, annual_return=0.0)
    assert r["future_value"] == 2200.0
    assert r["total_invested"] == 2200.0
    assert r["total_gain"] == 0.0


def test_lumpsum_compounds_to_annual_equivalent():
    # Monthly compounding reconstructs the annual figure; output is rounded to
    # paise, so allow a 1-paisa tolerance.
    r = project_growth(initial=1000, monthly=0, years=10, annual_return=0.10)
    assert math.isclose(r["future_value"], 1000 * 1.10**10, abs_tol=0.01)


def test_sip_grows_above_invested():
    r = project_growth(initial=0, monthly=10000, years=12, annual_return=0.12)
    assert r["future_value"] > r["total_invested"] > 0
    assert len(r["trajectory"]) == 12


def test_trajectory_is_monotonic_for_positive_return():
    r = project_growth(initial=5000, monthly=2000, years=5, annual_return=0.08)
    values = [point["value"] for point in r["trajectory"]]
    assert values == sorted(values)
