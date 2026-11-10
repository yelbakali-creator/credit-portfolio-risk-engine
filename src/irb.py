"""Basel IRB risk-weight function for corporate exposures and the Vasicek (ASRF) model.

The IRB formula is the 99.9% quantile of the loss of an infinitely granular portfolio
driven by a single systematic factor (Vasicek, 2002; Gordy, 2003):
    K = [LGD * N((G(PD) + sqrt(R) G(0.999)) / sqrt(1 - R)) - PD * LGD] * MA(M)
    R = 0.12 w + 0.24 (1 - w),  w = (1 - exp(-50 PD)) / (1 - exp(-50))
    MA(M) = (1 + (M - 2.5) b) / (1 - 1.5 b),  b = (0.11852 - 0.05478 ln PD)^2
    (MA = 1 at M = 1 year; Basel's default effective maturity is 2.5 years)
    RWA = 12.5 x K x EAD
"""
import numpy as np
from scipy.stats import norm


def irb_correlation(pd_):
    w = (1 - np.exp(-50 * np.asarray(pd_))) / (1 - np.exp(-50))
    return 0.12 * w + 0.24 * (1 - w)


def maturity_adjustment(pd_, maturity):
    b = (0.11852 - 0.05478 * np.log(np.asarray(pd_))) ** 2
    return (1 + (np.asarray(maturity) - 2.5) * b) / (1 - 1.5 * b)


def vasicek_quantile(pd_, rho, alpha):
    """alpha-quantile of the default rate of an infinitely granular portfolio."""
    return norm.cdf((norm.ppf(pd_) + np.sqrt(rho) * norm.ppf(alpha)) / np.sqrt(1 - rho))


def vasicek_cdf(x, pd_, rho):
    """Distribution function of the default rate in the Vasicek model."""
    return norm.cdf((np.sqrt(1 - rho) * norm.ppf(x) - norm.ppf(pd_)) / np.sqrt(rho))


def irb_capital(pd_, lgd, maturity, alpha=0.999):
    """IRB capital requirement K per unit of EAD (unexpected loss, maturity-adjusted)."""
    rho = irb_correlation(pd_)
    return lgd * (vasicek_quantile(pd_, rho, alpha) - pd_) * maturity_adjustment(pd_, maturity)
