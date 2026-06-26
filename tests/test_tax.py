from finplan.tools.tax import apply_ltcg_tax


def test_ltcg_on_a_normal_gain():
    # gain = 10,00,000 - 6,00,000 = 4,00,000; taxable = 4,00,000 - 1,25,000 =
    # 2,75,000; tax = 12.5% of 2,75,000 = 34,375; post-tax = 9,65,625.
    r = apply_ltcg_tax(1_000_000, 600_000)
    assert r["total_gain"] == 400_000
    assert r["taxable_gain"] == 275_000
    assert r["estimated_tax"] == 34_375
    assert r["post_tax_corpus"] == 965_625


def test_gain_below_exemption_is_untaxed():
    # gain of 1,00,000 is under the ₹1.25L exemption -> no tax.
    r = apply_ltcg_tax(700_000, 600_000)
    assert r["taxable_gain"] == 0
    assert r["estimated_tax"] == 0
    assert r["post_tax_corpus"] == 700_000


def test_a_loss_is_untaxed():
    r = apply_ltcg_tax(500_000, 600_000)
    assert r["total_gain"] == 0
    assert r["estimated_tax"] == 0
    assert r["post_tax_corpus"] == 500_000
