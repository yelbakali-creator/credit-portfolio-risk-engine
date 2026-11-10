"""Validation tests: Basel IRB formula, Vasicek limit, expected loss and Euler additivity."""
import numpy as np

from src.irb import irb_capital, irb_correlation, maturity_adjustment, vasicek_cdf, vasicek_quantile
from src.simulation import build_portfolio, homogeneous_portfolio_losses, simulate_losses


def test_irb_matches_published_worked_example():
    # PD 1.5%, LGD 40%, M = 3 years: R = 0.1768 and K = 7.98% (example rounds intermediate steps)
    assert abs(irb_correlation(0.015) - 0.1768) < 2e-4
    assert abs(irb_capital(0.015, 0.40, 3.0) - 0.0798) < 2e-4


def test_maturity_adjustment_is_one_at_one_year():
    assert np.allclose(maturity_adjustment(np.array([0.001, 0.01, 0.1]), 1.0), 1.0)


def test_vasicek_quantile_inverts_cdf():
    q = vasicek_quantile(0.01, 0.2, 0.999)
    assert abs(vasicek_cdf(q, 0.01, 0.2) - 0.999) < 1e-10


def test_large_homogeneous_portfolio_converges_to_vasicek():
    pd_, rho = 0.01, float(irb_correlation(0.01))
    dr = homogeneous_portfolio_losses(10_000, pd_, rho, 1_000_000, seed=7)
    assert abs(np.quantile(dr, 0.999) / vasicek_quantile(pd_, rho, 0.999) - 1) < 0.03


def test_expected_loss_and_euler_additivity():
    port = build_portfolio(n=300, seed=2)
    res = simulate_losses(port, 40_000, w=0.6, seed=3)
    el = (port["pd"] * port["lgd"] * port["ead"]).sum()
    se = res["loss"].std(ddof=1) / np.sqrt(len(res["loss"]))
    assert abs(res["loss"].mean() - el) < 4 * se
    v = np.quantile(res["loss"], 0.99)
    tail = res["loss"] >= v
    assert np.isclose(res["by_sector"][tail].mean(axis=0).sum(), res["loss"][tail].mean())
