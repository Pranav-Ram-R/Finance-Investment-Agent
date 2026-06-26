import pytest

from finplan.parsing import format_inr, parse_amount


def test_lakh_and_crore_words():
    assert parse_amount("50 lakh") == 5_000_000
    assert parse_amount("2 lakh") == 200_000
    assert parse_amount("1.5 crore") == 15_000_000
    assert parse_amount("1 crore") == 10_000_000


def test_plain_numbers_and_symbols():
    assert parse_amount("15000") == 15_000
    assert parse_amount("₹2,00,000") == 200_000
    assert parse_amount("50k") == 50_000


def test_passthrough_numbers():
    assert parse_amount(500000) == 500_000.0
    assert parse_amount(15000.0) == 15_000.0


def test_unparseable_raises():
    with pytest.raises(ValueError):
        parse_amount("a lot of money")


def test_format_inr_indian_grouping():
    assert format_inr(4_851_170) == "₹48,51,170"
    assert format_inr(10_060_982) == "₹1,00,60,982"
    assert format_inr(15_000) == "₹15,000"
    assert format_inr(148_830) == "₹1,48,830"
