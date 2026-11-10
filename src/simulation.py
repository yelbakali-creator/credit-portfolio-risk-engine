"""Credit portfolio construction and Monte Carlo simulation of default losses.

Asset return of obligor i in sector s:
    Y_i = sqrt(R_i) * S_s + sqrt(1 - R_i) * eps_i,   S_s = sqrt(w) Z_global + sqrt(1 - w) Z_s
R_i is the Basel IRB correlation, so each obligor carries the same systematic risk as in the
regulatory formula. With w = 1 there is a single factor (the Basel ASRF setting); with w < 1,
sectors are only partly correlated and the portfolio diversifies across sectors.
For the t copula, Y_i is divided by sqrt(W / nu) with W ~ chi2(nu) common to all obligors,
which creates joint extreme defaults (tail dependence).
Default if Y_i falls below the threshold matching PD_i. Simulation is done conditionally on
the factors: given them, defaults are independent with probability p_i(Z).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

from .irb import irb_correlation

MASTER_SCALE = {"AAA": 0.0003, "AA": 0.0005, "A": 0.0010, "BBB": 0.0030,
                "BB": 0.0120, "B": 0.0450, "CCC": 0.1500}
RATING_MIX = {"AAA": 0.03, "AA": 0.07, "A": 0.20, "BBB": 0.35, "BB": 0.20, "B": 0.12, "CCC": 0.03}
SECTORS = ["Energy", "Financials", "Industrials", "Consumer", "Technology", "Healthcare", "Real estate", "Utilities"]


def build_portfolio(n=2000, seed=11, lgd=0.45):
    """Illustrative corporate loan book: rating mix, sectors and concentrated exposures."""
    rng = np.random.default_rng(seed)
    ratings = rng.choice(list(RATING_MIX), size=n, p=list(RATING_MIX.values()))
    sector_w = np.array([0.14, 0.18, 0.16, 0.14, 0.12, 0.10, 0.10, 0.06])
    sectors = rng.choice(len(SECTORS), size=n, p=sector_w)
    ead = np.round(rng.lognormal(np.log(5e6), 1.1, n), -3)
    pd_ = np.array([MASTER_SCALE[r] for r in ratings])
    return pd.DataFrame({"obligor": np.arange(n), "rating": ratings, "sector": sectors,
                         "sector_name": [SECTORS[s] for s in sectors], "ead": ead, "pd": pd_,
                         "lgd": lgd, "rho": irb_correlation(pd_)})


def simulate_losses(port, n_scen, w=0.6, nu=None, seed=1, chunk=5000, keep_obligor_tail=None):
    """Portfolio loss per scenario, plus loss by sector and by rating.

    keep_obligor_tail: a loss threshold; if given, returns the sum of each obligor's loss over
    scenarios whose total loss is at or above the threshold (for Euler contributions).
    """
    rng = np.random.default_rng(seed)
    n = len(port)
    ead_lgd = (port["ead"] * port["lgd"]).values
    rho, sec = port["rho"].values, port["sector"].values
    n_sec = sec.max() + 1
    thresh = norm.ppf(port["pd"].values) if nu is None else student_t.ppf(port["pd"].values, nu)
    ratings = pd.Categorical(port["rating"], categories=list(MASTER_SCALE)).codes
    losses = np.empty(n_scen)
    by_sector = np.empty((n_scen, n_sec))
    by_rating = np.empty((n_scen, len(MASTER_SCALE)))
    tail_sum = np.zeros(n) if keep_obligor_tail is not None else None
    n_tail = 0
    for start in range(0, n_scen, chunk):
        m = min(chunk, n_scen - start)
        zg = rng.standard_normal((m, 1))
        zs = rng.standard_normal((m, n_sec))
        S = np.sqrt(w) * zg + np.sqrt(1 - w) * zs[:, sec]                 # m x n
        scale = 1.0 if nu is None else np.sqrt(rng.chisquare(nu, (m, 1)) / nu)
        p_cond = norm.cdf((thresh[None, :] * scale - np.sqrt(rho)[None, :] * S) / np.sqrt(1 - rho)[None, :])
        default = rng.random((m, n)) < p_cond
        L = default * ead_lgd[None, :]
        losses[start:start + m] = L.sum(axis=1)
        by_sector[start:start + m] = np.stack([L[:, sec == s].sum(axis=1) for s in range(n_sec)], axis=1)
        by_rating[start:start + m] = np.stack([L[:, ratings == r].sum(axis=1) for r in range(len(MASTER_SCALE))], axis=1)
        if keep_obligor_tail is not None:
            tail = losses[start:start + m] >= keep_obligor_tail
            tail_sum += L[tail].sum(axis=0)
            n_tail += tail.sum()
    out = {"loss": losses, "by_sector": by_sector, "by_rating": by_rating}
    if keep_obligor_tail is not None:
        out.update({"obligor_tail_sum": tail_sum, "n_tail": n_tail})
    return out


def homogeneous_portfolio_losses(n_obligors, pd_, rho, n_scen, seed=3):
    """Default rate of a homogeneous single-factor portfolio, simulated exactly via the
    conditional binomial distribution (used to test convergence to the Vasicek limit)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_scen)
    p = norm.cdf((norm.ppf(pd_) - np.sqrt(rho) * z) / np.sqrt(1 - rho))
    return rng.binomial(n_obligors, p) / n_obligors
