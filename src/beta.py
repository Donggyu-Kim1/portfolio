"""
베타 계산 (1년 윈도우, 일간 로그수익률 기준)
- 입력:
    config/portfolio.csv       (code,name,shares,price)
    data/raw/{name}.csv        (date, close, [change])
    data/raw/KOSPI.csv         (date, close, [change])
- 출력(append):
    data/result/portfolio_status_{YYYY-MM-DD}.md
      섹션: "## Beta (1Y, 일간 로그수익률)"
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# KST 날짜
try:
    from zoneinfo import ZoneInfo  # py>=3.9
except Exception:  # pragma: no cover
    ZoneInfo = None

# ===== 경로/상수 =====
PORTFOLIO_CSV = Path("config/portfolio.csv")
RAW_DIR = Path("data/raw")
RESULT_DIR = Path("data/result")
BENCHMARK_NAME = "KOSPI"  # data/raw/KOSPI.csv
MIN_OBS = 60  # 베타 계산 최소 관측치 수(부족하면 NaN 처리)


# ===== 유틸 =====
def now_kst_date() -> date:
    if ZoneInfo is not None:
        return datetime.now(ZoneInfo("Asia/Seoul")).date()
    return datetime.now().date()


def ensure_dirs():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


def fmt_pct(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "-"
    return f"{x * 100:.2f}%"


def fmt_num(x: Optional[float], digits: int = 4) -> str:
    if x is None or not np.isfinite(x):
        return "-"
    return f"{x:.{digits}f}"


def load_timeseries(name: str) -> pd.DataFrame:
    """
    data/raw/{name}.csv 로드: date, close, change 열 보정.
    change 없거나 NaN 포함 시 close로부터 로그수익률 재계산.
    """
    fp = RAW_DIR / f"{name}.csv"
    if not fp.exists():
        raise FileNotFoundError(f"시계열 파일이 없습니다: {fp}")

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

    # change 재계산(없거나 일부 NaN)
    if "change" not in df.columns or df["change"].isna().any():
        df["change"] = np.log(df["close"] / df["close"].shift(1))

    return df


def load_portfolio() -> pd.DataFrame:
    df = pd.read_csv(PORTFOLIO_CSV, dtype={"code": str, "name": str})
    required = {"code", "name", "shares", "price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"portfolio.csv에 필요한 컬럼이 없습니다: {missing}")
    df["code"] = df["code"].astype(str).str.strip().str.zfill(6)
    df["name"] = df["name"].astype(str).str.strip()
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df = df[(df["name"] != "") & (df["shares"] > 0) & (df["price"] > 0)].reset_index(
        drop=True
    )
    return df


def one_year_window(today: date) -> Tuple[date, date]:
    # "지난 1년"을 캘린더 365일로 정의
    try:
        from dateutil.relativedelta import relativedelta  # optional

        start = today - relativedelta(years=1)
    except Exception:
        start = today - timedelta(days=365)
    return (start, today)


def filter_window(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    return df[(df["date"] >= start) & (df["date"] <= end)].copy()


def beta_from_series(r_i: pd.Series, r_b: pd.Series) -> float:
    """
    베타 = Cov(r_i, r_b) / Var(r_b), 표본(ddof=1).
    관측치 부족/Var=0이면 np.nan 반환.
    """
    # 정렬/정합은 호출자에서 보장
    x = pd.Series(r_i, dtype=float)
    y = pd.Series(r_b, dtype=float)
    if len(x) < MIN_OBS or len(y) < MIN_OBS:
        return float("nan")
    var_b = y.var(ddof=1)
    if not np.isfinite(var_b) or var_b <= 0:
        return float("nan")
    cov_ib = np.cov(x, y, ddof=1)[0, 1]
    return float(cov_ib / var_b)


def intersection_dates(dfs: List[pd.DataFrame]) -> List[date]:
    """
    주어진 여러 DF의 date 교집합(오름차순 리스트) 반환
    """
    common = None
    for df in dfs:
        dset = set(df["date"].tolist())
        common = dset if common is None else (common & dset)
    if not common:
        return []
    return sorted(common)


def portfolio_returns_on_dates(
    pf: pd.DataFrame, price_map: Dict[str, pd.DataFrame], dates: List[date]
) -> pd.Series:
    """
    지정된 dates에서 포트폴리오 가치 V_t를 계산하고
    일간 로그수익률 R_p,t = ln(V_t / V_{t-1}) 반환.
    """
    # 각 name의 date->close 매핑 준비(빠른 접근)
    close_map: Dict[str, Dict[date, float]] = {}
    for name, df in price_map.items():
        close_map[name] = dict(zip(df["date"], df["close"]))

    V = []
    for d in dates:
        v = 0.0
        for _, row in pf.iterrows():
            name = row["name"]
            sh = float(row["shares"])
            c = close_map[name].get(d, np.nan)
            if not np.isfinite(c):
                v = np.nan
                break
            v += sh * c
        V.append(v)

    V = pd.Series(V, index=pd.Index(dates, name="date"), dtype=float)
    V = V.dropna()
    Rp = np.log(V / V.shift(1)).dropna()
    return Rp


# ===== 메인 =====
def main():
    ensure_dirs()
    asof = now_kst_date()
    start_1y, end_1y = one_year_window(asof)

    # 1) 데이터 로드
    pf = load_portfolio()
    bench_df = load_timeseries(BENCHMARK_NAME)
    bench_w = filter_window(bench_df, start_1y, end_1y)

    price_map: Dict[str, pd.DataFrame] = {}
    for name in pf["name"].tolist():
        ts = load_timeseries(name)
        price_map[name] = filter_window(ts, start_1y, end_1y)

    # 2) 개별 종목 베타 계산 (KOSPI와 날짜 교집합 기준)
    per_stock_rows = []
    for _, row in pf.iterrows():
        name = row["name"]
        code = row["code"]
        s_df = price_map[name]

        merged = pd.merge(
            s_df[["date", "change"]],
            bench_w[["date", "change"]].rename(columns={"change": "change_b"}),
            on="date",
            how="inner",
        ).dropna()

        n_obs = len(merged)
        beta_i = beta_from_series(merged["change"], merged["change_b"])
        period_txt = (
            f"{merged['date'].iloc[0]} ~ {merged['date'].iloc[-1]}"
            if n_obs > 0
            else "-"
        )

        per_stock_rows.append(
            {
                "name": name,
                "code": code,
                "n": n_obs,
                "period": period_txt,
                "beta": beta_i,
            }
        )

    per_stock_df = (
        pd.DataFrame(per_stock_rows).sort_values("name").reset_index(drop=True)
    )

    # 3) 포트폴리오 베타 (직접법)
    #    모든 종목과 벤치마크의 공통 날짜(1년 윈도우 내)에서 포트 가치 수익률 vs 벤치마크 수익률
    dfs_for_intersection = [bench_w] + [price_map[name] for name in pf["name"].tolist()]
    common_dates = intersection_dates(dfs_for_intersection)
    # 수익률은 연속 날짜가 2개 이상 필요
    if len(common_dates) >= (MIN_OBS + 1):  # 대략 관측치 MIN_OBS개 확보
        Rp = portfolio_returns_on_dates(pf, price_map, common_dates)
        Rb = bench_w.set_index("date").loc[Rp.index, "change"]
        n_p = len(Rp)
        beta_p = beta_from_series(Rp, Rb)
        p_period_txt = f"{Rp.index[0]} ~ {Rp.index[-1]}" if n_p > 0 else "-"
    else:
        Rp = pd.Series(dtype=float)
        Rb = pd.Series(dtype=float)
        n_p = 0
        beta_p = float("nan")
        p_period_txt = "-"

    # 4) Markdown 파일에 Append
    out_fp = RESULT_DIR / f"portfolio_status_{asof.isoformat()}.md"
    lines: List[str] = []
    lines.append("")
    lines.append("## Beta (1Y, 일간 로그수익률)")
    lines.append(f"- 기준일: **{asof}**, 윈도우: **{start_1y} ~ {end_1y}**")
    lines.append("")
    lines.append("### 포트폴리오 Beta")
    lines.append(f"- 관측치 수: **{n_p}**  (공통 거래일 기준)")
    lines.append(f"- 기간: **{p_period_txt}**")
    lines.append(f"- **Beta (직접법)**: **{fmt_num(beta_p, 4)}**")
    lines.append("")
    lines.append("### 개별 종목 Beta")
    lines.append("| name | code | n | period | beta |")
    lines.append("|---|---:|---:|---|---:|")
    for _, r in per_stock_df.iterrows():
        lines.append(
            f"| {r['name']} | {r['code']} | {int(r['n'])} | {r['period']} | {fmt_num(r['beta'], 4)} |"
        )
    lines.append("")

    # 파일이 있으면 추가, 없으면 생성
    mode = "a" if out_fp.exists() else "w"
    with open(out_fp, mode, encoding="utf-8") as w:
        # 파일 처음 생성 시 제목 헤더 넣어주고 싶다면 여기에 조건부 추가 가능
        w.write("\n".join(lines))

    print(f"[OK] Beta 결과 추가 완료: {out_fp}")


if __name__ == "__main__":
    main()
