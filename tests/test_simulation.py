from finplan.tools.simulation import monte_carlo_simulation


def test_percentiles_are_ordered():
    r = monte_carlo_simulation(100_000, 10_000, 10, 0.10, 0.15, seed=1)
    assert r["p10"] < r["p25"] < r["median"] < r["p75"] < r["p90"]


def test_probability_in_unit_interval():
    r = monte_carlo_simulation(100_000, 10_000, 10, 0.10, 0.15, goal=2_000_000, seed=1)
    assert 0.0 <= r["probability_of_reaching_goal"] <= 1.0


def test_trivially_low_goal_is_certain():
    r = monte_carlo_simulation(100_000, 10_000, 10, 0.10, 0.15, goal=1, seed=1)
    assert r["probability_of_reaching_goal"] == 1.0


def test_impossible_goal_is_near_zero():
    r = monte_carlo_simulation(100_000, 10_000, 10, 0.10, 0.15, goal=10**12, seed=1)
    assert r["probability_of_reaching_goal"] == 0.0
