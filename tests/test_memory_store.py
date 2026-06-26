from finplan.memory import store


def test_save_and_get_plan(tmp_path):
    db = tmp_path / "m.sqlite"
    store.save_plan(
        "u1", initial=200_000, monthly=15_000, years=12, goal=5_000_000,
        profile="moderate", allocation={"equity": 60, "debt": 30, "gold": 10},
        expected_return=0.106, projected_value=4_851_170, db_path=db,
    )
    plan = store.get_plan("u1", db_path=db)
    assert plan["goal"] == 5_000_000
    assert plan["allocation"]["equity"] == 60  # JSON round-trips to a dict


def test_save_plan_upserts(tmp_path):
    db = tmp_path / "m.sqlite"
    common = dict(allocation={"equity": 60, "debt": 30, "gold": 10}, expected_return=0.1)
    store.save_plan("u1", initial=1, monthly=1, years=1, goal=1, profile="conservative",
                    projected_value=2, db_path=db, **common)
    store.save_plan("u1", initial=2, monthly=2, years=2, goal=999, profile="moderate",
                    projected_value=4, db_path=db, **common)
    plan = store.get_plan("u1", db_path=db)
    assert plan["goal"] == 999 and plan["profile"] == "moderate"


def test_progress_log_is_ordered(tmp_path):
    db = tmp_path / "m.sqlite"
    store.log_contribution("u1", 30_000, 31_000, db_path=db)
    store.log_contribution("u1", 45_000, 47_000, db_path=db)
    history = store.get_progress("u1", db_path=db)
    assert len(history) == 2
    assert history[-1]["current_value"] == 47_000


def test_missing_plan_returns_none(tmp_path):
    db = tmp_path / "m.sqlite"
    assert store.get_plan("nobody", db_path=db) is None
