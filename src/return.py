"""
포트폴리오 수익률 계산 (오늘 기준, Markdown 출력)

입력:
  - config/portfolio.csv (columns: code,name,shares,price)
  - data/raw/{name}.csv  (columns: date, close, volume?, change?)  # change 없으면 close로 계산
  - data/raw/KOSPI.csv   (columns: date, close, volume?, change?)  # change 없으면 close로 계산

출력:
  - data/result/portfolio_returns_{YYYY-MM-DD}.md

계산 규칙 (기준일 = 오늘, KST):
  1) 종목 누적수익률 r_i_cum = ln( close_last_i / price_i )
  2) 포트폴리오 누적수익률 R_p_cum = ln( Σ_i shares_i*close_last_i / Σ_i shares_i*price_i )
  3) 종목 일일수익률 r_i_daily = 각 파일 마지막 행 change (없으면 ln(close/prev_close))
  4) 포트폴리오 일일수익률 R_p_daily = ln( V_t / V_{t-1} ), 공통 거래일에서 마지막 두 날짜로 계산
  5) 초과수익률 alpha_daily = R_p_daily - change_KOSPI (KOSPI change는 공통 마지막 날짜의 값)
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# KST 날짜
try:
    from zoneinfo import ZoneInfo  # py>=3.9
except Exception:  # pragma: no cover
    ZoneInfo = None

# ===== 경로 설정 =====
PORTFOLIO_CSV = Path("config/portfolio.csv")
RAW_DIR = Path("data/raw")
RESULT_DIR = Path("data/result")
BENCHMARK_NAME = "KOSPI"  # data/raw/KOSPI.csv


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


def fmt_num(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "-"
    return f"{x:,.0f}"


def load_timeseries(name: str) -> pd.DataFrame:
    """
    data/raw/{name}.csv 로드하여 date, close, change 열 보정.
    change 열이 없거나 일부 NaN이면 close로부터 재계산.
    """
    fp = RAW_DIR / f"{name}.csv"
    if not fp.exists():
        raise FileNotFoundError(f"시계열 파일이 없습니다: {fp}")

    df = pd.read_csv(fp)
    # 필요한 열 존재 보정
    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError(f"{fp}에 date/close 컬럼이 없습니다.")
    df = df[["date", "close"] + (["change"] if "change" in df.columns else [])].copy()

    # 타입/정렬
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    if "change" in df.columns:
        df["change"] = pd.to_numeric(df["change"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).drop_duplicates(
        subset=["date"], keep="last"
    )
    df = df.sort_values("date").reset_index(drop=True)

    # change 재계산(없거나 일부 NaN이면)
    if "change" not in df.columns or df["change"].isna().any():
        df["change"] = np.log(df["close"] / df["close"].shift(1))

    return df


def load_portfolio() -> pd.DataFrame:
    df = pd.read_csv(PORTFOLIO_CSV, dtype={"code": str, "name": str})
    required = {"code", "name", "shares", "price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"portfolio.csv에 필요한 컬럼이 없습니다: {missing}")
    # 정리
    df["code"] = df["code"].astype(str).str.strip().str.zfill(6)
    df["name"] = df["name"].astype(str).str.strip()
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    # 비정상 행 제거
    df = df[(df["name"] != "") & (df["shares"] > 0) & (df["price"] > 0)].reset_index(
        drop=True
    )
    return df


def latest_common_dates(price_map: Dict[str, pd.DataFrame]) -> Tuple[date, date]:
    """
    모든 종목에 공통으로 존재하는 '마지막 날짜'와 그 '이전 날짜'를 반환.
    (포트폴리오 일일 수익률 계산용)
    """
    # 각 종목의 날짜 집합 교집합
    common_dates = None
    for name, df in price_map.items():
        dset = set(df["date"].tolist())
        common_dates = dset if common_dates is None else (common_dates & dset)
    if not common_dates:
        raise ValueError("공통 거래일이 없습니다(모든 종목 날짜 교집합이 비어있음).")

    common_sorted = sorted(common_dates)
    if len(common_sorted) < 2:
        raise ValueError("공통 거래일이 2일 미만입니다(일일 수익률 계산 불가).")

    last_date = common_sorted[-1]
    prev_date = common_sorted[-2]
    return last_date, prev_date


# ===== 메인 로직 =====
def main():
    ensure_dirs()
    today = now_kst_date()

    # 1) 포트폴리오/벤치마크 로드
    pf = load_portfolio()
    bench_df = load_timeseries(BENCHMARK_NAME)

    # 2) 각 종목 시계열 로드
    price_map: Dict[str, pd.DataFrame] = {}
    for name in pf["name"].tolist():
        price_map[name] = load_timeseries(name)

    # 3) 개별 종목: 오늘(마지막 가용일) 기준 종가, 일일수익률(change), 누적수익률 계산
    rows = []
    V0 = 0.0  # 초기(매수 원가) 포트 가치
    Vt_any = 0.0  # 오늘 포트 가치 (개별 종목의 '각자 마지막 종가' 기준)

    for _, row in pf.iterrows():
        code = row["code"]
        name = row["name"]
        sh = float(row["shares"])
        price0 = float(row["price"])

        df = price_map[name]
        last_date = df["date"].iloc[-1]
        close_last = float(df["close"].iloc[-1])
        daily_change = (
            float(df["change"].iloc[-1])
            if np.isfinite(df["change"].iloc[-1])
            else np.nan
        )

        # 누적수익률 (오늘 기준): ln(close_last / price0)
        r_cum = (
            np.log(close_last / price0)
            if (price0 > 0 and np.isfinite(close_last))
            else np.nan
        )

        # 포트 가치 합산(누적용)
        V0 += sh * price0
        Vt_any += sh * close_last

        rows.append(
            {
                "name": name,
                "code": code,
                "shares": sh,
                "buy_price": price0,
                "last_date": last_date,
                "close_last": close_last,
                "r_i_cum": r_cum,
                "r_i_daily": daily_change,
            }
        )

    # 4) 포트폴리오 누적 수익률 (오늘 기준)
    R_p_cum = np.log(Vt_any / V0) if (V0 > 0 and np.isfinite(Vt_any)) else np.nan

    # 5) 포트폴리오 일일 수익률 (공통 거래일의 마지막 2일)
    last_common, prev_common = latest_common_dates(price_map)

    def close_on(df: pd.DataFrame, d: date) -> float:
        s = df.loc[df["date"] == d, "close"]
        if s.empty:
            raise ValueError(f"요청 일자 {d}의 가격이 없습니다.")
        return float(s.iloc[0])

    V_t = 0.0
    V_tm1 = 0.0
    for _, row in pf.iterrows():
        name = row["name"]
        sh = float(row["shares"])
        df = price_map[name]
        V_t += sh * close_on(df, last_common)
        V_tm1 += sh * close_on(df, prev_common)

    R_p_daily = np.log(V_t / V_tm1) if (V_tm1 > 0 and np.isfinite(V_t)) else np.nan

    # 6) 벤치마크(KOSPI) 일일 수익률: 공통 마지막 날짜의 change
    #    (없으면 가장 가까운 이전 날짜 사용)
    bench_on_last = bench_df.loc[bench_df["date"] == last_common]
    if bench_on_last.empty:
        # 가장 가까운 이전 날짜
        bench_df2 = bench_df[bench_df["date"] < last_common]
        if bench_df2.empty:
            raise ValueError("KOSPI에 적합한 비교 날짜가 없습니다.")
        bench_row = bench_df2.iloc[-1]
    else:
        bench_row = bench_on_last.iloc[0]

    R_b_daily = (
        float(bench_row["change"]) if np.isfinite(bench_row["change"]) else np.nan
    )
    bench_date_used = bench_row["date"]

    # 7) 초과수익률
    alpha_daily = (
        R_p_daily - R_b_daily
        if (np.isfinite(R_p_daily) and np.isfinite(R_b_daily))
        else np.nan
    )

    # 8) Markdown 출력
    asof = today.isoformat()
    out_fp = RESULT_DIR / f"portfolio_status_{asof}.md"

    # 개별 종목 표용 DataFrame
    report_df = pd.DataFrame(rows)
    report_df = report_df.sort_values("name").reset_index(drop=True)

    # 포맷팅용 열 추가
    report_df["buy_price_f"] = report_df["buy_price"].apply(fmt_num)
    report_df["close_last_f"] = report_df["close_last"].apply(fmt_num)
    report_df["r_i_cum_f"] = report_df["r_i_cum"].apply(fmt_pct)
    report_df["r_i_daily_f"] = report_df["r_i_daily"].apply(fmt_pct)
    report_df["last_date_f"] = report_df["last_date"].astype(str)

    # Markdown 구성
    lines: List[str] = []
    lines.append(f"# 포트폴리오 보고서 (기준일: {asof})")
    lines.append("")
    lines.append("## 요약")
    lines.append(
        f"- 포트폴리오 **누적 수익률**: **{fmt_pct(R_p_cum)}**  (V₀={fmt_num(V0)}, Vₜ={fmt_num(Vt_any)})"
    )
    lines.append(
        f"- 포트폴리오 **일일 수익률**: **{fmt_pct(R_p_daily)}**  (공통 거래일: {prev_common} → {last_common})"
    )
    lines.append(
        f"- KOSPI **일일 수익률**: **{fmt_pct(R_b_daily)}**  (비교 기준일: {bench_date_used})"
    )
    lines.append(f"- **초과수익률(일일)**: **{fmt_pct(alpha_daily)}**")
    lines.append("")
    lines.append("## 개별 종목 현황 (오늘 기준)")
    lines.append("")
    lines.append(
        "| name | code | shares | buy_price | last_date | close_last | r_i_cum | r_i_daily |"
    )
    lines.append("|---|---:|---:|---:|---|---:|---:|---:|")
    for _, r in report_df.iterrows():
        lines.append(
            f"| {r['name']} | {r['code']} | {int(r['shares'])} | {r['buy_price_f']} | {r['last_date_f']} | "
            f"{r['close_last_f']} | {r['r_i_cum_f']} | {r['r_i_daily_f']} |"
        )
    lines.append("")

    with open(out_fp, "w", encoding="utf-8") as w:
        w.write("\n".join(lines))

    print(f"[OK] 결과 저장: {out_fp}")


if __name__ == "__main__":
    main()
