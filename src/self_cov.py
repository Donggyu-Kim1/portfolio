#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ACF(1~9) & 10일 스케일 계수 계산 (최근 1년)
- 입력:
    config/portfolio.csv  (columns: code,name,shares,price)
    data/raw/{name}.csv   (columns: date, close, [change])
- 출력:
    data/processed/acf_1y_lag1_9_{YYYY-MM-DD}.csv
      컬럼: name, code, n_obs, start_date, end_date, rho_1..rho_9, nw_term, scale_10d
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional
import math

import numpy as np
import pandas as pd

# ===== 경로 설정 =====
PORTFOLIO_CSV = Path("config/portfolio.csv")
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
LOG_DIR = Path("data/_logs")
LOG_FILE = LOG_DIR / "acf_calc.log"

# KST 날짜
try:
    from zoneinfo import ZoneInfo  # py>=3.9
except Exception:  # pragma: no cover
    ZoneInfo = None


# ===== 유틸 =====
def now_kst_date() -> date:
    if ZoneInfo is not None:
        return datetime.now(ZoneInfo("Asia/Seoul")).date()
    return datetime.now().date()


def ensure_dirs():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def safe_log(msg: str) -> None:
    ensure_dirs()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as w:
        w.write(f"[{ts}] {msg}\n")


def one_year_window(today: date) -> Tuple[date, date]:
    try:
        from dateutil.relativedelta import relativedelta

        start = today - relativedelta(years=1)
    except Exception:
        start = today - timedelta(days=365)
    return start, today


def load_portfolio() -> pd.DataFrame:
    df = pd.read_csv(PORTFOLIO_CSV, dtype={"code": str, "name": str})
    need = {"code", "name", "shares", "price"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"portfolio.csv에 필요한 컬럼이 없습니다: {miss}")
    df["code"] = df["code"].astype(str).str.strip().str.zfill(6)
    df["name"] = df["name"].astype(str).str.strip()
    # 보유수량/매수가 검사만, 계산엔 직접 사용 안 함
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df = df[(df["name"] != "")].reset_index(drop=True)
    return df


def load_timeseries(name: str) -> pd.DataFrame:
    """
    data/raw/{name}.csv → date, close, change 정리.
    change 없거나 NaN이면 close로부터 ln 수익률 재계산.
    """
    fp = RAW_DIR / f"{name}.csv"
    if not fp.exists():
        raise FileNotFoundError(f"시계열 파일 없음: {fp}")

    df = pd.read_csv(fp)
    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError(f"{fp}에 date/close 컬럼이 없습니다.")

    use_cols = ["date", "close"] + (["change"] if "change" in df.columns else [])
    df = df[use_cols].copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    if "change" in df.columns:
        df["change"] = pd.to_numeric(df["change"], errors="coerce")

    df = df.dropna(subset=["date", "close"]).drop_duplicates(
        subset=["date"], keep="last"
    )
    df = df.sort_values("date").reset_index(drop=True)

    # change 보정
    if "change" not in df.columns or df["change"].isna().any():
        df["change"] = np.log(df["close"] / df["close"].shift(1))

    return df


def filter_window(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    return df[(df["date"] >= start) & (df["date"] <= end)].copy()


def acf_lags(r: pd.Series, max_lag: int = 9) -> List[float]:
    """
    표본 자기상관 ρ_k (k=1..max_lag). 길이가 lag 이하이면 NaN.
    """
    r = pd.Series(r).dropna().astype(float)
    out: List[float] = []
    n = len(r)
    for k in range(1, max_lag + 1):
        if n > k:
            try:
                rho = float(r.autocorr(lag=k))
            except Exception:
                rho = float("nan")
        else:
            rho = float("nan")
        out.append(rho)
    return out


def nw_term_from_rhos(rhos: List[float]) -> float:
    """
    T = 1 + 2 * sum_{k=1..9} (1 - k/10) * rho_k
    수치 안정화: T < 0 이면 0으로 클리핑
    """
    T = 1.0
    for k, rho in enumerate(rhos, start=1):
        if np.isfinite(rho):
            T += 2.0 * (1.0 - k / 10.0) * float(rho)
    if not np.isfinite(T):
        return float("nan")
    return max(T, 0.0)


def scale_10d_from_T(T: float) -> float:
    if not np.isfinite(T):
        return float("nan")
    return math.sqrt(10.0) * math.sqrt(T)


# ===== 메인 =====
def main():
    ensure_dirs()
    asof = now_kst_date()
    start_1y, end_1y = one_year_window(asof)

    pf = load_portfolio()

    rows: List[dict] = []
    for _, row in pf.iterrows():
        name = row["name"]
        code = row["code"]

        try:
            ts = load_timeseries(name)
            ts_w = filter_window(ts, start_1y, end_1y)
            # returns
            r = ts_w["change"].dropna()
            n_obs = int(len(r))
            if len(ts_w) >= 1:
                start_date = ts_w["date"].iloc[0]
                end_date = ts_w["date"].iloc[-1]
            else:
                start_date = None
                end_date = None

            # 최소 표본 체크(권장: 60 이상). 부족해도 가능한 범위로 계산, lag>n이면 NaN.
            rhos = acf_lags(r, max_lag=9)
            T = nw_term_from_rhos(rhos)
            s10 = scale_10d_from_T(T)

            row_out = {
                "name": name,
                "code": code,
                "n_obs": n_obs,
                "start_date": str(start_date) if start_date else "-",
                "end_date": str(end_date) if end_date else "-",
                "nw_term": T,
                "scale_10d": s10,
            }
            for i, rho in enumerate(rhos, start=1):
                row_out[f"rho_{i}"] = rho

            rows.append(row_out)

        except Exception as e:
            safe_log(f"[ERROR] {name}({code}) ACF 계산 실패: {e}")
            # 실패 시 NaN으로 기록
            fail_row = {
                "name": name,
                "code": code,
                "n_obs": 0,
                "start_date": "-",
                "end_date": "-",
                "nw_term": float("nan"),
                "scale_10d": float("nan"),
            }
            for i in range(1, 10):
                fail_row[f"rho_{i}"] = float("nan")
            rows.append(fail_row)

    df_out = pd.DataFrame(rows)
    # 컬럼 순서 정리
    base_cols = ["name", "code", "n_obs", "start_date", "end_date"]
    rho_cols = [f"rho_{i}" for i in range(1, 10)]
    tail_cols = ["nw_term", "scale_10d"]
    df_out = (
        df_out[base_cols + rho_cols + tail_cols]
        .sort_values(["name", "code"])
        .reset_index(drop=True)
    )

    out_fp = PROCESSED_DIR / f"acf_1y_lag1_9_{asof.isoformat()}.csv"
    df_out.to_csv(out_fp, index=False, encoding="utf-8")
    print(f"[OK] 저장 완료: {out_fp}")


if __name__ == "__main__":
    main()
