"""Tax tool — post-tax corpus under Indian Long-Term Capital Gains (LTCG) rules.

Like every figure in this project, the tax is computed in Python (never by the
LLM). We model the equity LTCG rule in force from FY2024-25: gains on equity held
over a year are taxed at 12.5%, with the first ₹1.25 lakh of gains exempt.

This is a deliberate simplification — a single equity rate over the whole gain —
because equity drives most of a long-horizon corpus. The rate and exemption are
parameters, so a what-if can override them. It is an estimate, not tax advice.
"""
from __future__ import annotations

# Equity LTCG, FY2024-25 onward.
LTCG_RATE = 0.125          # 12.5% on long-term equity gains
LTCG_EXEMPTION = 125_000   # first ₹1.25 lakh of gains is exempt


def apply_ltcg_tax(
    corpus: float,
    total_invested: float,
    ltcg_rate: float = LTCG_RATE,
    exemption: float = LTCG_EXEMPTION,
) -> dict:
    """Estimate LTCG tax on redemption and the resulting post-tax corpus.

    The gain is ``corpus - total_invested``; the exemption is subtracted before
    applying the rate. Both the gain and the taxable gain are floored at zero so
    a loss (or a sub-exemption gain) produces no tax.
    """
    gain = max(0.0, corpus - total_invested)
    taxable_gain = max(0.0, gain - exemption)
    tax = taxable_gain * ltcg_rate

    return {
        "pre_tax_corpus": round(corpus, 2),
        "total_invested": round(total_invested, 2),
        "total_gain": round(gain, 2),
        "exemption": exemption,
        "ltcg_rate": ltcg_rate,
        "taxable_gain": round(taxable_gain, 2),
        "estimated_tax": round(tax, 2),
        "post_tax_corpus": round(corpus - tax, 2),
    }
