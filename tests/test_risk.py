from finplan.tools.risk import (
    assess_risk_profile,
    blended_portfolio_stats,
    recommend_allocation,
)


def test_long_horizon_high_tolerance_is_aggressive():
    assert assess_risk_profile(12, "high")["profile"] == "aggressive"


def test_short_horizon_caps_profile_to_conservative():
    # Horizon limits capacity even when the user says high tolerance.
    assert assess_risk_profile(1, "high")["profile"] == "conservative"


def test_low_tolerance_overrides_long_horizon():
    # Willingness limits risk even when the horizon could support more.
    assert assess_risk_profile(15, "low")["profile"] == "conservative"


def test_allocation_always_sums_to_100():
    for profile in ("conservative", "moderate", "aggressive"):
        alloc = recommend_allocation(profile, 10)["allocation"]
        assert sum(alloc.values()) == 100


def test_short_horizon_caps_equity_and_rebalances():
    alloc = recommend_allocation("aggressive", 2)["allocation"]
    assert alloc["equity"] == 30
    assert sum(alloc.values()) == 100


def test_blended_return_is_weighted_average():
    allocation = {"equity": 60, "debt": 30, "gold": 10}
    stats = {
        "equity": {"cagr": 0.12, "volatility": 0.15},
        "debt": {"cagr": 0.07, "volatility": 0.03},
        "gold": {"cagr": 0.08, "volatility": 0.12},
    }
    out = blended_portfolio_stats(allocation, stats)
    assert out["expected_return"] == round(0.6 * 0.12 + 0.3 * 0.07 + 0.1 * 0.08, 4)


def test_tolerance_synonyms_are_accepted():
    # The LLM/user may say "moderate" / "aggressive" rather than our enum.
    assert assess_risk_profile(12, "moderate")["willingness"] == "moderate"
    assert assess_risk_profile(12, "aggressive")["willingness"] == "aggressive"
    assert assess_risk_profile(12, "Conservative")["willingness"] == "conservative"


def test_allocation_accepts_profile_synonyms():
    assert (
        recommend_allocation("moderate", 10)["allocation"]
        == recommend_allocation("medium", 10)["allocation"]
    )
