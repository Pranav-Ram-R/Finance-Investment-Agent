from finplan.config import DEFAULTS, describe_config, resolve_spec


def test_preset_resolves_to_full_spec():
    assert resolve_spec("llama-70b") == "groq:llama-3.3-70b-versatile"


def test_full_spec_passes_through():
    assert resolve_spec("groq:some-model") == "groq:some-model"


def test_describe_config_covers_all_roles():
    cfg = describe_config()
    assert set(cfg) == set(DEFAULTS)
    assert all(":" in spec for spec in cfg.values())
