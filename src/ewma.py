"""
EWMA 변동성 (1년, λ=0.94, Daily) - 개별 종목만
- 입력:
    config/portfolio.csv  (code,name,shares,price)
    data/raw/{name}.csv   (date, close, [change])
- 출력(append):
    data/result/portfolio_status_{YYYY-MM-DD}.md
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# ===== 설정 =====
PORTFOLIO_CSV = Path("config/portfolio.csv")
RAW_DIR = Path("data/raw")
RESULT_DIR = Path("data/result")
PROCESSED_DIR = Path("data/processed")
LAMBDA = 0.94
INIT_WINDOW = 60  # 초기 분산 추정을 위한 최소 구간(부족하면 가능한 범위 사용)

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
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def fmt_num(x: Optional[float], digits: int = 6) -> str:
    if x is None or not np.isfinite(x):
        return "-"
    return f"{x:.{digits}f}"


def load_portfolio() -> pd.DataFrame:
    df = pd.read_csv(PORTFOLIO_CSV, dtype={"code": str, "name": str})
    need = {"code", "name", "shares", "price"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"portfolio.csv에 필요한 컬럼이 없습니다: {miss}")
    df["code"] = df["code"].astype(str).str.strip().str.zfill(6)
    df["name"] = df["name"].astype(str).str.strip()
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df = df[(df["name"] != "") & (df["shares"] > 0) & (df["price"] > 0)].reset_index(
        drop=True
    )
    return df


def load_timeseries(name: str) -> pd.DataFrame:
    """
    data/raw/{name}.csv → date, close, change 정리.
    change가 없거나 NaN이면 close로부터 ln 수익률 계산.
    """
    fp = RAW_DIR / f"{name}.csv"
    if not fp.exists():
        raise FileNotFoundError(f"시계열 파일 없음: {fp}")

    df = pd.read_csv(fp)
    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError(f"{fp}에 date/close 컬럼이 없습니다.")

    use_cols = ["date", "close"]
    if "change" in df.columns:
        use_cols.append("change")
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


def one_year_window(today: date) -> Tuple[date, date]:
    try:
        from dateutil.relativedelta import relativedelta  # optional

        start = today - relativedelta(years=1)
    except Exception:
        start = today - timedelta(days=365)
    return (start, today)


def filter_window(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    return df[(df["date"] >= start) & (df["date"] <= end)].copy()


def ewma_sigma_daily(
    returns: pd.Series, lam: float = LAMBDA, init_window: int = INIT_WINDOW
) -> float:
    """
    returns: 일간 로그수익률 Series (NaN 제거되어 있어야 함)
    - 초기 분산: 앞쪽 min(init_window, N) 구간 표본분산(ddof=1)
    - 재귀: σ_t^2 = lam*σ_{t-1}^2 + (1-lam)*r_t^2
    - 반환: 마지막 σ_t (일간)
    """
    r = pd.Series(returns.dropna().astype(float))
    n = len(r)
    if n < 2:
        return float("nan")

    m = min(init_window, n)
    if m >= 2:
        sigma2 = float(r.iloc[:m].var(ddof=1))
        idx_start = m
    else:
        sigma2 = float(r.iloc[0] ** 2)
        idx_start = 1

    for i in range(idx_start, n):
        ri = float(r.iloc[i])
        sigma2 = lam * sigma2 + (1.0 - lam) * (ri * ri)

    return float(np.sqrt(sigma2))


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
        except Exception as e:
            # 개별 종목 실패 시 스킵하되 행은 남김
            rows.append(
                {
                    "name": name,
                    "code": code,
                    "n": 0,
                    "period": "-",
                    "ewma_sigma_daily": float("nan"),
                    "error": str(e),
                }
            )
            continue

        ts_w = filter_window(ts, start_1y, end_1y)
        # 관측치(수익률) 개수: 마지막-첫번째 기준으로 -1
        ret = ts_w["change"].dropna()
        n_obs = int(len(ret))
        if len(ts_w) >= 1:
            period_txt = f"{ts_w['date'].iloc[0]} ~ {ts_w['date'].iloc[-1]}"
        else:
            period_txt = "-"

        sigma = ewma_sigma_daily(ret, lam=LAMBDA, init_window=INIT_WINDOW)

        rows.append(
            {
                "name": name,
                "code": code,
                "n": n_obs,
                "period": period_txt,
                "ewma_sigma_daily": sigma,
                "error": "",
            }
        )

    df_out = pd.DataFrame(rows).sort_values("name").reset_index(drop=True)

    csv_fp = (
        PROCESSED_DIR
        / f"ewma_1y_lambda{str(LAMBDA).replace('.', '_')}_{asof.isoformat()}.csv"
    )
    df_out.to_csv(csv_fp, index=False, encoding="utf-8")
    print(f"[OK] EWMA CSV 저장: {csv_fp}")

    # Markdown append
    out_fp = RESULT_DIR / f"portfolio_status_{asof.isoformat()}.md"
    lines: List[str] = []
    lines.append("")
    lines.append(f"## EWMA Vol (1Y, λ={LAMBDA}, Daily)")
    lines.append(f"- 기준일: **{asof}**, 윈도우: **{start_1y} ~ {end_1y}**")
    lines.append("- 결과는 **일간 표준편차(σ)**이며 **연간화하지 않음**")
    lines.append("")
    lines.append("| name | code | n | period | ewma_sigma_daily |")
    lines.append("|---|---:|---:|---|---:|")
    for _, r in df_out.iterrows():
        lines.append(
            f"| {r['name']} | {r['code']} | {int(r['n'])} | {r['period']} | {fmt_num(r['ewma_sigma_daily'], 6)} |"
        )
    lines.append("")

    mode = "a" if out_fp.exists() else "w"
    with open(out_fp, mode, encoding="utf-8") as w:
        w.write("\n".join(lines))

    print(f"[OK] EWMA 결과 추가 완료: {out_fp}")


if __name__ == "__main__":
    main()
