"""
5Y rolling backtest of 10-day 99% VaR using EWMA(λ=0.94) + ACF scaling + rolling 1Y correlations
+ Tail calibration (expanding, no look-ahead)
+ Christoffersen independence & conditional coverage tests

Inputs:
  - config/portfolio.csv           (name, shares)
  - data/raw/{name}.csv            (date, close)
Outputs:
  - data/result/var_backtest_portfolio_{start}_{end}.csv
  - data/result/var_backtest_portfolio_{start}_{end}.md
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import math
import numpy as np
import pandas as pd

# ===== User settings =====
BASE_DIR = Path(".")
PORTFOLIO_CSV = BASE_DIR / "config" / "portfolio.csv"
RAW_DIR = BASE_DIR / "data" / "raw"
RESULT_DIR = BASE_DIR / "data" / "result"

WINDOW_DAYS = 252  # rolling window for risk estimation (~1y)
HORIZON_DAYS = 10  # 10 trading days
LAMBDA = 0.94  # EWMA lambda
Z_99 = 2.33  # 99% VaR z (normal)
MIN_CALIB_SAMPLES = 252  # 최소 표본 쌓일 때까지는 캘리브=1.0 사용


# ===== Helpers =====
def _load_prices(names: List[str]) -> pd.DataFrame:
    """Load close prices for given names as a wide DataFrame indexed by date."""
    frames = []
    for name in names:
        fp = RAW_DIR / f"{name}.csv"
        if not fp.exists():
            print(f"[경고] price file missing: {fp}")
            continue
        df = pd.read_csv(fp)
        cols = {c.lower().strip(): c for c in df.columns}
        if "date" not in cols or "close" not in cols:
            print(f"[경고] missing required columns in {fp.name}")
            continue
        d = df[[cols["date"], cols["close"]]].copy()
        d.columns = ["date", name]
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = (
            d.dropna(subset=["date", name])
            .drop_duplicates(subset=["date"])
            .sort_values("date")
        )
        frames.append(d)
    if not frames:
        raise RuntimeError("No valid raw price files loaded.")
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="date", how="inner")  # strict intersection
    out = out.sort_values("date").reset_index(drop=True)
    return out


def _log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from price wide frame (index: date col, columns: names)."""
    r = np.log(prices / prices.shift(1))
    return r.dropna(how="all")


def _ewma_sigma_daily(window_r: pd.DataFrame, lam: float) -> pd.Series:
    """RiskMetrics EWMA sigma (per series) using the last WINDOW point."""
    m = len(window_r)
    weights = lam ** np.arange(m - 1, -1, -1)  # oldest->newest
    wnorm = 1 - lam
    vals = {}
    for c in window_r.columns:
        x = window_r[c].to_numpy()
        if np.all(np.isnan(x)):
            vals[c] = np.nan
            continue
        x = np.nan_to_num(x, nan=0.0)
        s2 = wnorm * float(np.sum(weights * (x**2)))
        vals[c] = math.sqrt(max(s2, 0.0))
    return pd.Series(vals)


def _acf_scale_10d(window_r: pd.DataFrame, max_lag: int = 9, n: int = 10) -> pd.Series:
    """
    Estimate ACF up to max_lag and return scale_10d = sqrt( n + 2*sum_{k=1}^{max_lag} (n-k) * rho_k )
    per series, based on window returns.
    """

    def acf_one(x: np.ndarray, lag: int) -> float:
        x = x - np.nanmean(x)
        num = np.nansum(x[lag:] * x[:-lag])
        den = np.nansum(x * x)
        if den == 0 or np.isnan(den):
            return 0.0
        return float(num / den)

    scales = {}
    for c in window_r.columns:
        x = window_r[c].astype(float).to_numpy()
        if np.sum(np.isfinite(x)) < max_lag + 2:
            scales[c] = math.sqrt(n)  # fallback to sqrt(10)
            continue
        rhos = [acf_one(x, k) for k in range(1, max_lag + 1)]
        var_factor = n + 2 * sum((n - k) * rhos[k - 1] for k in range(1, max_lag + 1))
        var_factor = max(var_factor, 0.0)
        scales[c] = math.sqrt(var_factor)
    return pd.Series(scales)


def _near_psd_corr(C: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Make a correlation matrix near-PSD by eigenvalue clipping."""
    C = (C + C.T) / 2.0
    vals, vecs = np.linalg.eigh(C)
    vals = np.clip(vals, eps, None)
    C_psd = (vecs * vals) @ vecs.T
    d = np.sqrt(np.clip(np.diag(C_psd), eps, None))
    Cn = C_psd / d[:, None] / d[None, :]
    np.fill_diagonal(Cn, 1.0)
    return Cn


def _kupiec_pof(n_exceed: int, n_obs: int, alpha: float = 0.01) -> Tuple[float, float]:
    """
    Kupiec Proportion of Failures test (unconditional coverage).
    Returns (LR_uc, p_value). df=1 -> p = 1 - erf(sqrt(LR/2))
    """
    if n_obs == 0:
        return float("nan"), float("nan")
    pi_hat = n_exceed / n_obs if n_obs > 0 else 0.0
    pi_hat = min(max(pi_hat, 1e-12), 1 - 1e-12)
    alpha = min(max(alpha, 1e-12), 1 - 1e-12)
    logL_pi = (n_exceed * math.log(pi_hat)) + (
        (n_obs - n_exceed) * math.log(1 - pi_hat)
    )
    logL_a = (n_exceed * math.log(alpha)) + ((n_obs - n_exceed) * math.log(1 - alpha))
    LR = -2.0 * (logL_a - logL_pi)
    pval = 1.0 - math.erf(math.sqrt(max(LR, 0.0) / 2.0))
    return LR, pval


def _christoffersen_tests(
    exceeds: np.ndarray, alpha: float = 0.01
) -> Tuple[float, float, float, float]:
    """
    Christoffersen independence & conditional coverage tests.
    Returns: (LR_ind, p_ind, LR_cc, p_cc)
    """
    x = exceeds.astype(int)
    if x.size < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")

    n00 = int(np.sum((x[:-1] == 0) & (x[1:] == 0)))
    n01 = int(np.sum((x[:-1] == 0) & (x[1:] == 1)))
    n10 = int(np.sum((x[:-1] == 1) & (x[1:] == 0)))
    n11 = int(np.sum((x[:-1] == 1) & (x[1:] == 1)))
    n0 = n00 + n01
    n1 = n10 + n11

    # transition probabilities
    pi01 = (n01 / n0) if n0 else 0.0
    pi11 = (n11 / n1) if n1 else 0.0
    pi = (n01 + n11) / (n0 + n1) if (n0 + n1) else 0.0

    def _ll(p, n1, n0):
        p = min(max(p, 1e-12), 1 - 1e-12)
        return n1 * math.log(p) + n0 * math.log(1 - p)

    # LR for independence (H0: pi01 = pi11 = pi)
    LL_h0 = _ll(pi, n01 + n11, n00 + n10)
    LL_h1 = _ll(pi01, n01, n00) + _ll(pi11, n11, n10)
    LR_ind = -2.0 * (LL_h0 - LL_h1)
    p_ind = 1.0 - math.erf(math.sqrt(max(LR_ind, 0.0) / 2.0))  # ~Chi^2(1)

    # Conditional coverage = Kupiec (unconditional) + independence
    LR_uc, _ = _kupiec_pof(int(x.sum()), int(len(x)), alpha=alpha)
    LR_cc = LR_uc + LR_ind
    p_cc = 1.0 - math.erf(math.sqrt(max(LR_cc, 0.0) / 2.0))  # ~Chi^2(2)

    return LR_ind, p_ind, LR_cc, p_cc


# ===== Main backtest =====
def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load portfolio
    pf = pd.read_csv(PORTFOLIO_CSV)
    if "name" not in pf.columns or "shares" not in pf.columns:
        raise ValueError("config/portfolio.csv must contain columns: name, shares")
    names = [str(x).strip() for x in pf["name"].tolist()]
    shares = pd.Series(pf["shares"].astype(float).values, index=names)

    # 2) Load prices (inner-join on dates for clean matrix)
    prices = _load_prices(names)
    prices = prices.set_index("date")

    # 3) Portfolio value series
    shares_vec = shares.reindex(prices.columns)
    if shares_vec.isna().any():
        missing = shares_vec[shares_vec.isna()].index.tolist()
        raise ValueError(f"Shares missing for: {missing}")
    V = (prices * shares_vec).sum(axis=1)

    # 4) Log returns matrix
    rets = _log_returns(prices)
    if len(rets) < WINDOW_DAYS + HORIZON_DAYS + 1:
        raise RuntimeError(
            "Not enough history to run the backtest with given window/horizon."
        )

    rows = []
    u_history: List[float] = []  # standardized losses history for expanding calibration

    for t_end in range(WINDOW_DAYS, len(rets) - HORIZON_DAYS):
        # rolling window & dates
        r_window = rets.iloc[t_end - WINDOW_DAYS : t_end]
        date_t = rets.index[t_end]
        date_t10 = rets.index[t_end + HORIZON_DAYS]

        # 4-1) EWMA daily sigma per name
        sigma_daily = _ewma_sigma_daily(r_window, LAMBDA)

        # 4-2) ACF-based scale_10d per name
        scale_10d = _acf_scale_10d(r_window, max_lag=9, n=HORIZON_DAYS)

        # 4-3) 10d sigma per name
        sigma_10d = sigma_daily * scale_10d

        # 4-4) correlation over window
        R = r_window.corr().fillna(0.0).to_numpy()
        R = _near_psd_corr(R)

        # 4-5) exposures at time t (use prices at date_t)
        p_t = prices.loc[date_t]
        x = (p_t * shares_vec).to_numpy(dtype=float)

        # align arrays (already aligned by columns order)
        sigma_vec = sigma_10d.reindex(prices.columns).to_numpy(dtype=float)

        # joint validity
        valid_mask = np.isfinite(x) & np.isfinite(sigma_vec)
        if valid_mask.sum() < 2:
            continue
        x_v = x[valid_mask]
        s_v = sigma_vec[valid_mask]
        R_v = R[np.ix_(valid_mask, valid_mask)]

        # 4-6) portfolio variance in money (10d)
        Sigma_10d = (s_v[:, None] * R_v) * s_v[None, :]
        var_money = float(x_v @ Sigma_10d @ x_v)
        port_sigma_money = math.sqrt(max(var_money, 0.0))
        var_99 = Z_99 * port_sigma_money

        # --- Tail calibration (expanding, no look-ahead) ---
        if len(u_history) >= MIN_CALIB_SAMPLES:
            z_hat = float(pd.Series(u_history).quantile(0.99))
            calib_factor_t = (z_hat / Z_99) if z_hat > 0 else 1.0
        else:
            calib_factor_t = 1.0
        var_99_calib = calib_factor_t * var_99

        # 4-7) realized 10d loss
        V_t = float(V.loc[date_t])
        V_t10 = float(V.loc[date_t10])
        realized_loss = max(V_t - V_t10, 0.0)

        exceed = 1 if realized_loss > var_99 else 0
        exceed_calib = 1 if realized_loss > var_99_calib else 0

        # standardized loss for future calibration updates
        u_t = (realized_loss / port_sigma_money) if port_sigma_money > 0 else np.nan
        if np.isfinite(u_t):
            u_history.append(float(u_t))

        rows.append(
            {
                "date": date_t.date().isoformat(),
                "end_date": date_t10.date().isoformat(),
                "portfolio_value_t": V_t,
                "var_99_10d": var_99,
                "var_99_10d_calib": var_99_calib,
                "calib_factor_t": calib_factor_t,
                "realized_loss_10d": realized_loss,
                "exceed": exceed,
                "exceed_calib": exceed_calib,
                "n_names_used": int(valid_mask.sum()),
                "mean_sigma_daily": float(np.nanmean(sigma_daily.to_numpy())),
                "mean_scale_10d": float(np.nanmean(scale_10d.to_numpy())),
                "port_sigma_money_10d": port_sigma_money,
            }
        )

    if not rows:
        raise RuntimeError(
            "No backtest rows generated; check data coverage and window sizes."
        )

    bt = pd.DataFrame(rows)
    start_str = bt["date"].iloc[0]
    end_str = bt["date"].iloc[-1]
    out_csv = RESULT_DIR / f"var_backtest_portfolio_{start_str}_{end_str}.csv"
    bt.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # ---- Summary stats (raw & calibrated) ----
    def _summ(exceed_col: str, label: str) -> Tuple[str, str]:
        n_obs = int(bt.shape[0])
        n_exc = int(bt[exceed_col].sum())
        exc_rate = n_exc / n_obs if n_obs else float("nan")
        LR_uc, p_uc = _kupiec_pof(n_exc, n_obs, alpha=0.01)
        LR_ind, p_ind, LR_cc, p_cc = _christoffersen_tests(
            bt[exceed_col].to_numpy(), alpha=0.01
        )
        block = [
            f"### {label}",
            f"- Exceedances: **{n_exc} / {n_obs}**  (rate={exc_rate:.2%}, expected=1.00%)",
            f"- Kupiec (UC): LR={LR_uc:.3f}, p={p_uc:.4f}",
            f"- Christoffersen Independence: LR={LR_ind:.3f}, p={p_ind:.4f}",
            f"- Christoffersen Conditional Coverage: LR={LR_cc:.3f}, p={p_cc:.4f}",
        ]
        return "\n".join(block), (
            n_obs,
            n_exc,
            exc_rate,
            LR_uc,
            p_uc,
            LR_ind,
            p_ind,
            LR_cc,
            p_cc,
        )

    raw_md, _ = _summ("exceed", "Raw VaR (Normal z=2.33)")
    cal_md, _ = _summ("exceed_calib", "Calibrated VaR (expanding)")

    # Global recommended calibration (post-hoc, for reporting only)
    if len(u_history) > 0:
        z_hat_full = float(pd.Series(u_history).quantile(0.99))
        calib_full = z_hat_full / Z_99 if z_hat_full > 0 else 1.0
    else:
        z_hat_full, calib_full = float("nan"), 1.0

    md_lines = []
    md_lines.append(f"# VaR Backtest — Portfolio (10d, 99%)")
    md_lines.append(f"- Period: {start_str} → {end_str}  (obs={bt.shape[0]})")
    md_lines.append(
        f"- Mean(σ_daily) over windows: {bt['mean_sigma_daily'].mean():.4f}"
    )
    md_lines.append(
        f"- Mean(scale_10d) over windows: {bt['mean_scale_10d'].mean():.3f}"
    )
    md_lines.append(
        f"- Mean(port σ_10d, money): {bt['port_sigma_money_10d'].mean():,.0f} KRW"
    )
    md_lines.append("")
    md_lines.append(raw_md)
    md_lines.append("")
    md_lines.append(cal_md)
    md_lines.append("")
    md_lines.append("## Tail Calibration Summary")
    md_lines.append(
        f"- Expanding calibration used online with MIN_CALIB_SAMPLES={MIN_CALIB_SAMPLES}"
    )
    md_lines.append(f"- Post-hoc ẑ@99 from standardized losses: {z_hat_full:.3f}")
    md_lines.append(
        f"- Recommended global factor (reporting): **×{calib_full:.2f}**  (={z_hat_full:.3f} / 2.33)"
    )
    md_lines.append("")
    md_lines.append("## Notes")
    md_lines.append(
        "- EWMA λ=0.94, window=252d; ACF lags=1..9; correlation=rolling 1y; near-PSD enforced."
    )
    md_lines.append(
        "- Realized loss uses *10 trading days ahead* portfolio value (buy-and-hold shares)."
    )
    md_lines.append(
        "- Tail calibration multiplies VaR by the 99% quantile of standardized losses from *past* observations only."
    )
    md_lines.append(
        "- The global factor is post-hoc and for reporting; do not use it online without a split (train/test)."
    )

    out_md = RESULT_DIR / f"var_backtest_portfolio_{start_str}_{end_str}.md"
    with open(out_md, "w", encoding="utf-8") as w:
        w.write("\n".join(md_lines))

    print(f"[완료] Backtest CSV: {out_csv}")
    print(f"[완료] Backtest MD : {out_md}")
    print(
        f"[요약] Raw exceed={int(bt['exceed'].sum())}/{bt.shape[0]}  |  Calib exceed={int(bt['exceed_calib'].sum())}/{bt.shape[0]}"
    )
    print(f"[Calib] post-hoc z_hat@99={z_hat_full:.3f}  ->  factor ~ x{calib_full:.2f}")


if __name__ == "__main__":
    main()
