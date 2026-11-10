"""Credit portfolio risk study: Basel IRB vs Monte Carlo economic capital."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.irb import irb_capital, irb_correlation, vasicek_quantile
from src.simulation import (MASTER_SCALE, SECTORS, build_portfolio, homogeneous_portfolio_losses,
                            simulate_losses)

ROOT = Path(__file__).parent
TABLES, FIGS = ROOT / "results" / "tables", ROOT / "results" / "figures"
for folder in (TABLES, FIGS):
    folder.mkdir(parents=True, exist_ok=True)
NAVY, RED, GREY, GOLD, GREEN = "#1F3864", "#C0392B", "#7F8C8D", "#D4A017", "#2E7D32"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

N_SCEN, ALPHA, W_SECTOR, NU = 200_000, 0.999, 0.6, 5


def var_ci(loss, alpha, level=0.95):
    """Distribution-free confidence interval for a Monte Carlo quantile (order statistics)."""
    x = np.sort(loss)
    n = len(x)
    half = 1.96 * np.sqrt(n * alpha * (1 - alpha))
    return x[int(np.floor(n * alpha - half))], x[int(np.ceil(n * alpha + half))]


def var_es(loss, alpha):
    v = np.quantile(loss, alpha)
    return v, loss[loss >= v].mean()


def main():
    # 1. Validation of the IRB implementation against a published worked example
    k_example = irb_capital(0.015, 0.40, 3.0)
    irb_check = pd.DataFrame([
        ["Asset correlation R (PD 1.5%)", irb_correlation(0.015), 0.1768],
        ["Capital K (PD 1.5%, LGD 40%, M 3y)", k_example, 0.0798],
        ["RWA on EAD 50m (million)", k_example * 12.5 * 50, 49.9],
    ], columns=["quantity", "this_project", "published_example_rounded"])
    irb_check.to_csv(TABLES / "irb_formula_check.csv", index=False)

    # 2. Convergence to the Vasicek limit: homogeneous single-factor portfolios
    pd0 = 0.01
    rho0 = irb_correlation(pd0)
    limit = vasicek_quantile(pd0, rho0, ALPHA)
    conv = []
    for n in [50, 200, 1_000, 10_000, 100_000]:
        dr = homogeneous_portfolio_losses(n, pd0, rho0, 2_000_000, seed=n)
        q = np.quantile(dr, ALPHA)
        conv.append({"obligors": n, "mc_default_rate_99_9": q, "vasicek_limit": limit, "relative_gap": q / limit - 1})
    conv = pd.DataFrame(conv)
    conv.to_csv(TABLES / "vasicek_convergence.csv", index=False)

    # 3. Realistic portfolio
    port = build_portfolio()
    port.to_csv(TABLES / "portfolio.csv", index=False)
    ead = port["ead"].sum()
    el_analytic = (port["pd"] * port["lgd"] * port["ead"]).sum()
    # MA(M) equals 1 at M = 1 year: the like-for-like comparison with a one-year default model.
    # Basel's default effective maturity of 2.5 years adds a maturity (downgrade) adjustment on top.
    irb_k = irb_capital(port["pd"].values, port["lgd"].values, 1.0)
    irb_ul = (irb_k * port["ead"]).sum()
    irb_ul_m25 = (irb_capital(port["pd"].values, port["lgd"].values, 2.5) * port["ead"]).sum()
    hhi = ((port["ead"] / ead) ** 2).sum()

    runs = {
        "Single factor (Basel setting)": dict(w=1.0, nu=None, seed=21),
        "Multi-factor, 8 sectors": dict(w=W_SECTOR, nu=None, seed=22),
        "Multi-factor, t copula (nu = 5)": dict(w=W_SECTOR, nu=NU, seed=23),
    }
    sims, capital = {}, []
    for name, kw in runs.items():
        res = simulate_losses(port, N_SCEN, **kw)
        sims[name] = res
        v, es = var_es(res["loss"], ALPHA)
        capital.append({"model": name, "expected_loss_mc": res["loss"].mean(), "var_99_9": v, "es_99_9": es,
                        "economic_capital_var_minus_el": v - res["loss"].mean(),
                        "mc_se_expected_loss": res["loss"].std(ddof=1) / np.sqrt(N_SCEN)})
    capital = pd.DataFrame(capital)
    capital["irb_unexpected_loss_m1"] = irb_ul
    capital["irb_unexpected_loss_m2_5"] = irb_ul_m25
    capital["ratio_to_irb"] = capital["economic_capital_var_minus_el"] / irb_ul
    capital.to_csv(TABLES / "capital_comparison.csv", index=False)

    # Name concentration: same single-factor model on an equal-exposure version of the book
    granular = port.copy()
    granular["ead"] = ead / len(port)
    g = simulate_losses(granular, N_SCEN, w=1.0, seed=24)
    v_g, _ = var_es(g["loss"], ALPHA)
    irb_ul_g = (irb_k * granular["ead"]).sum()                          # M = 1 year
    lo_g, hi_g = var_ci(g["loss"], ALPHA)
    ci_g = ((lo_g - g["loss"].mean()) / irb_ul_g, (hi_g - g["loss"].mean()) / irb_ul_g)

    # 4. Euler allocation of ES (multi-factor Gaussian): contributions add up to total ES
    base = sims["Multi-factor, 8 sectors"]
    v_b, es_b = var_es(base["loss"], ALPHA)
    tail = base["loss"] >= v_b
    contrib_sector = pd.DataFrame({
        "sector": SECTORS, "exposure_share": [port.loc[port["sector"] == s, "ead"].sum() / ead for s in range(len(SECTORS))],
        "expected_loss": base["by_sector"].mean(axis=0), "es_contribution": base["by_sector"][tail].mean(axis=0)})
    contrib_sector["es_share"] = contrib_sector["es_contribution"] / es_b
    contrib_sector.to_csv(TABLES / "es_contributions_by_sector.csv", index=False)
    contrib_rating = pd.DataFrame({
        "rating": list(MASTER_SCALE), "exposure_share": [port.loc[port["rating"] == r, "ead"].sum() / ead for r in MASTER_SCALE],
        "es_contribution": base["by_rating"][tail].mean(axis=0)})
    contrib_rating["es_share"] = contrib_rating["es_contribution"] / es_b
    contrib_rating.to_csv(TABLES / "es_contributions_by_rating.csv", index=False)

    # Obligor-level contributions (second pass with the same seed)
    ob = simulate_losses(port, N_SCEN, w=W_SECTOR, seed=22, keep_obligor_tail=v_b)
    port["es_contribution"] = ob["obligor_tail_sum"] / ob["n_tail"]
    top = port.sort_values("es_contribution", ascending=False).head(10)
    top[["obligor", "rating", "sector_name", "ead", "pd", "es_contribution"]].to_csv(TABLES / "top10_es_contributors.csv", index=False)

    # 5. Stress tests (multi-factor Gaussian)
    stress = []
    for label, mult in [("Base", 1.0), ("PDs x 2 (downturn)", 2.0), ("PDs x 3 (severe)", 3.0)]:
        sp = port.copy()
        sp["pd"] = np.minimum(sp["pd"] * mult, 0.99)
        r = simulate_losses(sp, 100_000, w=W_SECTOR, seed=31)
        v, es = var_es(r["loss"], ALPHA)
        stress.append({"scenario": label, "expected_loss": r["loss"].mean(), "var_99_9": v, "es_99_9": es,
                       "irb_unexpected_loss_m1": (irb_capital(sp["pd"].values, sp["lgd"].values, 1.0) * sp["ead"]).sum()})
    stress = pd.DataFrame(stress)
    stress.to_csv(TABLES / "stress_tests.csv", index=False)

    validation = pd.DataFrame([
        ["Expected loss: analytic", el_analytic],
        ["Expected loss: Monte Carlo (multi-factor)", base["loss"].mean()],
        ["Monte Carlo standard error of EL", base["loss"].std(ddof=1) / np.sqrt(N_SCEN)],
        ["ES 99.9% (multi-factor)", es_b],
        ["Sum of sector ES contributions", contrib_sector["es_contribution"].sum()],
        ["Sum of obligor ES contributions", port["es_contribution"].sum()],
        ["Equal-exposure book: MC capital / IRB (granularity check)", (v_g - g["loss"].mean()) / irb_ul_g],
        ["Equal-exposure book: 95% CI of that ratio (lower)", ci_g[0]],
        ["Equal-exposure book: 95% CI of that ratio (upper)", ci_g[1]],
        ["Herfindahl index of exposures", hhi],
        ["Number of equivalent equal exposures (1 / HHI)", 1 / hhi],
    ], columns=["check", "value"])
    validation.to_csv(TABLES / "validation.csv", index=False)

    # 6. Charts
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.semilogx(conv["obligors"], conv["mc_default_rate_99_9"] * 100, "o-", color=NAVY, label="Monte Carlo, 99.9% default rate")
    ax.axhline(limit * 100, color=RED, ls="--", label=f"Vasicek / Basel limit: {limit:.2%}")
    ax.set_xlabel("Number of obligors (homogeneous portfolio, PD 1%)"); ax.set_ylabel("99.9% default rate (%)")
    ax.set_title("Granularity: Monte Carlo converges to the Basel formula")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "vasicek_convergence.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for (name, res), col in zip(sims.items(), [NAVY, GREEN, RED]):
        x = np.sort(res["loss"]) / 1e6
        ax.semilogy(x, 1 - np.arange(1, len(x) + 1) / (len(x) + 1), color=col, lw=1.2, label=name)
    ax.axhline(1 - ALPHA, color=GREY, ls=":", lw=1)
    ax.text(ax.get_xlim()[1] * 0.98, (1 - ALPHA) * 1.2, "99.9% level", ha="right", fontsize=7, color=GREY)
    ax.set_xlabel("Portfolio loss (million)"); ax.set_ylabel("Probability of exceeding (log scale)")
    ax.set_title("Loss distribution tails: dependence structure drives the 1-in-1000 loss")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "loss_tails.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ["Basel IRB\n(formula, M = 1y)", "Single factor\n(Monte Carlo)", "Multi-factor\n(8 sectors)", "t copula\n(nu = 5)"]
    vals = [irb_ul] + list(capital["economic_capital_var_minus_el"])
    ax.bar(labels, np.array(vals) / 1e6, color=[GREY, NAVY, GREEN, RED])
    for i, v in enumerate(vals):
        ax.text(i, v / 1e6, f"{v / 1e6:.0f}m", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Capital for unexpected loss at 99.9% (million)")
    ax.set_title("Regulatory vs economic capital")
    fig.tight_layout(); fig.savefig(FIGS / "capital_comparison.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    idx = np.arange(len(SECTORS))
    ax.bar(idx - 0.2, contrib_sector["exposure_share"] * 100, width=0.4, color=GREY, label="Share of exposure")
    ax.bar(idx + 0.2, contrib_sector["es_share"] * 100, width=0.4, color=NAVY, label="Share of ES 99.9% (Euler)")
    ax.set_xticks(idx, SECTORS, rotation=20)
    ax.set_ylabel("%"); ax.set_title("Risk contributions by sector vs exposure")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "es_contributions_by_sector.png", dpi=150); plt.close(fig)

    # Console summary
    pd.set_option("display.width", 200)
    print(f"Portfolio: {len(port)} obligors, EAD {ead / 1e9:.2f}bn, analytic EL {el_analytic / 1e6:.2f}m, "
          f"IRB UL {irb_ul / 1e6:.1f}m at M=1y ({irb_ul_m25 / 1e6:.1f}m at M=2.5y), HHI {hhi:.5f} (1/HHI = {1 / hhi:.0f})")
    print("\nIRB formula check\n" + irb_check.round(4).to_string(index=False))
    print("\nVasicek convergence\n" + conv.round(5).to_string(index=False))
    print("\nCapital comparison\n" + capital.round(0).to_string(index=False))
    print(f"\nEqual-exposure book (single factor): MC capital {(v_g - g['loss'].mean()) / 1e6:.1f}m vs IRB {irb_ul_g / 1e6:.1f}m")
    print("\nValidation\n" + validation.to_string(index=False))
    print("\nES by sector\n" + contrib_sector.round(3).to_string(index=False))
    print("\nES by rating\n" + contrib_rating.round(3).to_string(index=False))
    print("\nTop 10 contributors\n" + top[["obligor", "rating", "sector_name", "ead", "pd", "es_contribution"]].to_string(index=False))
    print("\nStress tests\n" + stress.round(0).to_string(index=False))


if __name__ == "__main__":
    main()
