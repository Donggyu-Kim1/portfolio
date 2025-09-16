"""
10-day 99% VaR (individual & portfolio) + Component VaR & 15% threshold check
Files used:
  - config/portfolio.csv               (name, shares[, ...])
  - data/raw/{name}.csv                (date, close, ...)
  - data/processed/ewma_1y_lambda{LAMBDA}_{asof}.csv        (name, ewma_sigma_daily)
  - data/processed/acf_1y_lag1_9_{asof}.csv                 (name, scale_10d)
  - data/processed/corr_1y_{asof}.csv                        (correlation matrix with index/cols=name)
Outputs:
  - data/result/var_individual_{asof}.csv
  - data/result/var_portfolio_{asof}.csv
  - data/result/portfolio_status_{asof}.md
"""

from __future__ import annotations

from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import math
import numpy as np
import pandas as pd

# ===== Settings =====
BASE_DIR = Path(".")
PORTFOLIO_CSV = BASE_DIR / "config" / "portfolio.csv"
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULT_DIR = BASE_DIR / "data" / "result"

LAMBDA = 0.94  # for ewma file name
Z_99 = 2.33  # 99% z-score per your spec
THRESHOLD_RATIO = 0.15  # 15% threshold vs total portfolio value


# ========= Helpers =========
def _norm_name(s: str) -> str:
    """Normalize security name keys for joining."""
    return (s or "").strip()


def _read_last_close_and_date(
    fp: Path,
) -> Tuple[Optional[pd.Timestamp], Optional[float]]:
    """
    Read last non-null date & close from data/raw/{name}.csv.
    Expected columns: date, close
    """
    if not fp.exists():
        return None, None
    try:
        df = pd.read_csv(fp)
        cols = {c.lower().strip(): c for c in df.columns}
        date_col = cols.get("date")
        close_col = cols.get("close")
        if date_col is None or close_col is None:
            return None, None
        df = df[[date_col, close_col]].dropna(subset=[close_col]).copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col)
        if df.empty:
            return None, None
        last_row = df.iloc[-1]
        return pd.Timestamp(last_row[date_col]).normalize(), float(last_row[close_col])
    except Exception:
        return None, None


def _most_common_date(dates: List[pd.Timestamp]) -> Optional[pd.Timestamp]:
    if not dates:
        return None
    cnt = Counter([d.date() for d in dates if pd.notna(d)])
    if not cnt:
        return None
    max_freq = max(cnt.values())
    candidates = [pd.Timestamp(d) for d, v in cnt.items() if v == max_freq]
    return max(candidates) if candidates else None


@dataclass
class PositionInfo:
    name: str
    shares: float
    close: float
    last_date: pd.Timestamp


# ========= Main =========
def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load portfolio (name, shares)
    pf = pd.read_csv(PORTFOLIO_CSV, dtype={"code": str})
    if "name" not in pf.columns or "shares" not in pf.columns:
        raise ValueError("config/portfolio.csv must contain columns: name, shares")
    pf["name_key"] = pf["name"].map(_norm_name)
    pf["shares"] = pd.to_numeric(pf["shares"], errors="coerce")
    pf = pf.dropna(subset=["name_key", "shares"])
    pf = pf[pf["shares"] != 0]

    # 2) Get last close & date for each name
    pos_rows: List[PositionInfo] = []
    missing_price: List[str] = []
    all_last_dates: List[pd.Timestamp] = []

    for _, row in pf.iterrows():
        name = str(row["name"])
        shares = float(row["shares"])
        raw_fp = RAW_DIR / f"{name}.csv"
        last_date, last_close = _read_last_close_and_date(raw_fp)
        if last_date is None or last_close is None or math.isnan(last_close):
            missing_price.append(name)
            continue
        pos_rows.append(
            PositionInfo(
                name=name, shares=shares, close=last_close, last_date=last_date
            )
        )
        all_last_dates.append(last_date)

    if not pos_rows:
        raise RuntimeError(
            "No positions with valid last close were found under data/raw/*.csv"
        )

    # 3) asof date = mode of last dates (latest among modes)
    asof = _most_common_date(all_last_dates)
    if asof is None:
        raise RuntimeError("Failed to determine asof date from raw files")
    asof_str = asof.date().isoformat()

    off_date = [p for p in pos_rows if p.last_date.normalize() != asof.normalize()]
    if off_date:
        print(
            f"[경고] 일부 종목의 마지막 날짜가 다릅니다. asof={asof_str} 로 통일합니다:"
        )
        for p in off_date[:10]:
            print(
                f"  - {p.name}: last_date={p.last_date.date().isoformat()} (≠ {asof_str})"
            )
        if len(off_date) > 10:
            print(f"  ... {len(off_date)-10} more")

    # 4) Load ewma & acf scalers and corr matrix
    ewma_fp = (
        PROCESSED_DIR / f"ewma_1y_lambda{str(LAMBDA).replace('.', '_')}_{asof_str}.csv"
    )
    acf_fp = PROCESSED_DIR / f"acf_1y_lag1_9_{asof_str}.csv"
    corr_fp = PROCESSED_DIR / f"corr_1y_{asof_str}.csv"  # end_date == asof

    if not ewma_fp.exists():
        raise FileNotFoundError(f"Missing file: {ewma_fp}")
    if not acf_fp.exists():
        raise FileNotFoundError(f"Missing file: {acf_fp}")
    if not corr_fp.exists():
        raise FileNotFoundError(f"Missing file: {corr_fp}")

    ewma = pd.read_csv(ewma_fp)
    acf = pd.read_csv(acf_fp)

    if "name" not in ewma.columns or "ewma_sigma_daily" not in ewma.columns:
        raise ValueError(f"{ewma_fp} must have columns: name, ewma_sigma_daily")
    if "name" not in acf.columns or "scale_10d" not in acf.columns:
        raise ValueError(f"{acf_fp} must have columns: name, scale_10d")

    ewma["name_key"] = ewma["name"].map(_norm_name)
    acf["name_key"] = acf["name"].map(_norm_name)

    ewma = ewma[["name_key", "ewma_sigma_daily"]].dropna()
    acf = acf[["name_key", "scale_10d"]].dropna()

    # 5) Build table (individuals)
    pos_df = pd.DataFrame(
        [
            {
                "name": p.name,
                "name_key": _norm_name(p.name),
                "shares": p.shares,
                "close": p.close,
                "position_value": p.shares * p.close,
                "last_date": p.last_date.date().isoformat(),
            }
            for p in pos_rows
        ]
    )

    merged = pos_df.merge(ewma, on="name_key", how="left").merge(
        acf, on="name_key", how="left"
    )

    merged["note"] = ""
    merged.loc[
        merged["ewma_sigma_daily"].isna(), "note"
    ] += "[missing ewma_sigma_daily]"
    merged.loc[merged["scale_10d"].isna(), "note"] += "[missing scale_10d]"

    merged["sigma_10d"] = merged["ewma_sigma_daily"] * merged["scale_10d"]
    merged["z"] = Z_99
    merged["var_individual"] = Z_99 * merged["position_value"] * merged["sigma_10d"]

    # initialize Component VaR columns (filled later for valid names)
    merged["cvar"] = np.nan
    merged["cvar_port_pct"] = np.nan  # share (%) of portfolio VaR
    merged["cvar_pos_pct"] = np.nan  # cVaR / position_value

    # keep only valid rows for portfolio calc
    valid = merged.dropna(subset=["position_value", "sigma_10d"])
    valid = valid[valid["position_value"] != 0]
    name_keys = valid["name_key"].tolist()

    # 6) Align correlation
    corr = pd.read_csv(corr_fp, index_col=0)
    corr.index = corr.index.map(_norm_name)
    corr.columns = [_norm_name(c) for c in corr.columns]

    missing_in_index = [k for k in name_keys if k not in corr.index]
    missing_in_cols = [k for k in name_keys if k not in corr.columns]
    if missing_in_index or missing_in_cols:
        print("[경고] 상관행렬에 없는 종목이 있어 포트폴리오 계산에서 제외합니다:")
        for k in sorted(set(missing_in_index + missing_in_cols)):
            print(f"  - {k}")

    present_keys = [k for k in name_keys if (k in corr.index and k in corr.columns)]
    valid = (
        valid.set_index("name_key").loc[present_keys].reset_index()
    )  # reorder = present_keys order
    if valid.empty:
        raise RuntimeError(
            "No overlapping names between positions and correlation matrix"
        )

    R = corr.loc[present_keys, present_keys].astype(float)
    np.fill_diagonal(R.values, 1.0)
    R = (R + R.T) / 2.0

    # vectors (aligned to present_keys)
    x = valid["position_value"].to_numpy(dtype=float)  # (n,)
    s = valid["sigma_10d"].to_numpy(dtype=float)  # (n,)

    # Σ = diag(σ) * R * diag(σ) (10d return covariance)
    Sigma = (s[:, None] * R.values) * s[None, :]
    var_money = float(x @ Sigma @ x)
    port_sigma_money = math.sqrt(max(var_money, 0.0))
    var_portfolio = Z_99 * port_sigma_money

    # ---- Component VaR (marginal * weight) ----
    # marginal VaR m = z * (Σ x) / σ_p
    if port_sigma_money > 0:
        Sx = Sigma @ x  # (n,)
        m = Z_99 * Sx / port_sigma_money  # (n,)
        cvar_vec = x * m  # (n,)
        # numerical guard: tiny negatives to zero
        cvar_vec = np.where(cvar_vec < 0, 0.0, cvar_vec)
    else:
        cvar_vec = np.zeros_like(x)

    # fill into merged for present_keys
    cvar_ser = pd.Series(cvar_vec, index=present_keys)
    # portfolio value used (sum of valid positions)
    portfolio_value_used = float(valid["position_value"].sum())
    # ratios
    cvar_port_pct = cvar_ser / var_portfolio if var_portfolio > 0 else np.nan
    cvar_pos_pct = cvar_ser / valid.set_index("name_key")["position_value"]

    # write back
    merged = merged.set_index("name_key")
    merged.loc[cvar_ser.index, "cvar"] = cvar_ser.values
    merged.loc[cvar_port_pct.index, "cvar_port_pct"] = cvar_port_pct.values
    merged.loc[cvar_pos_pct.index, "cvar_pos_pct"] = cvar_pos_pct.values
    merged = merged.reset_index()

    # portfolio-level ratio vs total portfolio value
    var_portfolio_ratio = (
        var_portfolio / portfolio_value_used if portfolio_value_used > 0 else np.nan
    )
    exceeds_threshold = (
        (var_portfolio_ratio > THRESHOLD_RATIO)
        if np.isfinite(var_portfolio_ratio)
        else False
    )

    # 7) Output CSVs
    indiv_out = merged[
        [
            "name",
            "name_key",
            "shares",
            "close",
            "position_value",
            "ewma_sigma_daily",
            "scale_10d",
            "sigma_10d",
            "z",
            "var_individual",
            "cvar",
            "cvar_port_pct",
            "cvar_pos_pct",
            "note",
            "last_date",
        ]
    ].sort_values("cvar", ascending=False, na_position="last")

    indiv_fp = RESULT_DIR / f"var_individual_{asof_str}.csv"
    indiv_out.to_csv(indiv_fp, index=False, encoding="utf-8-sig")

    port_out = pd.DataFrame(
        [
            {
                "asof": asof_str,
                "portfolio_value": portfolio_value_used,
                "z": Z_99,
                "portfolio_sigma_money": port_sigma_money,
                "var_portfolio": var_portfolio,
                "var_portfolio_ratio": var_portfolio_ratio,  # VaR / total portfolio value
                "threshold_15pct_exceeded": bool(exceeds_threshold),
                "names_used": len(valid),
                "names_total": len(merged),
                "names_missing_price": len(missing_price),
                "names_missing_in_corr": len(set(name_keys) - set(present_keys)),
            }
        ]
    )

    port_fp = RESULT_DIR / f"var_portfolio_{asof_str}.csv"
    port_out.to_csv(port_fp, index=False, encoding="utf-8-sig")

    # 8) MD summary
    md_lines = []
    md_lines.append("")
    md_lines.append(f"## 포트폴리오 VaR (10-day, 99%) — {asof_str}")
    md_lines.append("")
    md_lines.append(f"- Total portfolio value (used): {portfolio_value_used:,.0f} KRW")
    md_lines.append(f"- z-score: {Z_99}")
    md_lines.append(f"- Portfolio σ (money, 10d): {port_sigma_money:,.0f} KRW")
    md_lines.append(f"- **Portfolio VaR (10d, 99%)**: **{var_portfolio:,.0f} KRW**")
    if np.isfinite(var_portfolio_ratio):
        md_lines.append(
            f"- VaR / Portfolio value: **{var_portfolio_ratio:.2%}** "
            + ("→ **⚠️ 15% 초과**" if exceeds_threshold else "→ ✅ 15% 이하")
        )
    md_lines.append("")

    if missing_price:
        md_lines.append(
            f"> ⚠️ 종가 파일 누락/에러로 제외된 종목: {', '.join(missing_price[:10])}"
            + (" ..." if len(missing_price) > 10 else "")
        )
    missing_inputs = merged[merged["note"] != ""]
    if not missing_inputs.empty:
        names_miss = ", ".join(missing_inputs["name"].head(10).tolist())
        more = " ..." if len(missing_inputs) > 10 else ""
        md_lines.append(
            f"> ⚠️ 변동성/스케일 누락으로 제외 또는 부분계산: {names_miss}{more}"
        )
    not_in_corr = sorted(set(merged["name_key"]) - set(present_keys))
    if not_in_corr:
        md_lines.append(
            f"> ⚠️ 상관행렬에 없음(제외): {', '.join([n for n in not_in_corr][:10])}"
            + (" ..." if len(not_in_corr) > 10 else "")
        )

    # --- Component VaR section (Top 10) ---
    md_lines.append("## Top 10 Component VaR")
    md_lines.append("")
    topc = indiv_out.dropna(subset=["cvar"]).head(10)
    if not topc.empty:
        md_lines.append(
            "| Rank | Name | Position (KRW) | σ10d | cVaR (KRW) | cVaR/Port | cVaR/Pos | Note |"
        )
        md_lines.append("|---:|---|---:|---:|---:|---:|---:|---|")
        for i, r in enumerate(topc.itertuples(index=False), 1):
            md_lines.append(
                f"| {i} | {r.name} | {r.position_value:,.0f} | {r.sigma_10d:.4f} | {r.cvar:,.0f} | "
                f"{(r.cvar_port_pct if pd.notna(r.cvar_port_pct) else float('nan')):.2%} | "
                f"{(r.cvar_pos_pct  if pd.notna(r.cvar_pos_pct)  else float('nan')):.2%} | {r.note or ''} |"
            )
        # 합계 검증(표시용)
        total_cvar = float(np.nansum(indiv_out["cvar"].to_numpy()))
        md_lines.append("")
        md_lines.append(f"- Σ cVaR (all names) = {total_cvar:,.0f} KRW")
    else:
        md_lines.append("_No component VaR rows available_")

    md_fp = RESULT_DIR / f"portfolio_status_{asof_str}.md"
    mode = "a" if md_fp.exists() else "w"
    with open(md_fp, mode, encoding="utf-8") as w:
        w.write("\n".join(md_lines) + "\n")

    print(f"[완료] 개별 종목 VaR: {indiv_fp}")
    print(f"[완료] 포트폴리오 VaR 요약: {port_fp}")
    print(f"[완료] MD 요약: {md_fp}")


if __name__ == "__main__":
    main()
