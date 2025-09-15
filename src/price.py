"""
KRX 일별 종가/거래량 수집 (증분 + 자동 교정)
- 입력: config/portfolio.csv (컬럼명: code)
- 출력: data/raw/{ticker}.csv (컬럼: date, close, volume, change)
- 동작: 매 실행 시 기존 파일의 마지막 날짜(L)부터 '오늘'까지 다시 내려받아 덮어씀.
       (장중 임시 종가도 다음 실행에서 확정치로 자연 교정)
"""

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, date
from typing import List, Optional

import numpy as np
import pandas as pd

# Python 3.9+: 표준 라이브러리 시간대
try:
    from zoneinfo import ZoneInfo  # type: ignore
except Exception:
    ZoneInfo = None  # pragma: no cover

# pykrx
from pykrx import stock as krx

# ===== 설정 =====
PORTFOLIO_CSV = Path("config/portfolio.csv")
RAW_DIR = Path("data/raw")
LOG_DIR = Path("data/_logs")
LOG_FILE = LOG_DIR / "price_download.log"

# 최초 실행 시 기본으로 내려받는 기간(년)
DEFAULT_YEARS_BACK = 10


# ===== 유틸 =====
def now_kst_date() -> date:
    """오늘 날짜(KST) 반환."""
    if ZoneInfo is not None:
        return datetime.now(ZoneInfo("Asia/Seoul")).date()
    # ZoneInfo 미지원 환경 폴백 (로컬 타임존 사용)
    return datetime.now().date()


def yyyymmdd(d: date) -> str:
    return d.strftime("%Y%m%d")


def read_portfolio(fp: Path) -> pd.DataFrame:
    """config/portfolio.csv에서 code 컬럼을 읽어 6자리 문자열 리스트로 반환."""
    df = pd.read_csv(fp, dtype={"code": str})
    df["code"] = df["code"].str.zfill(6)
    return df[["code", "name"]]


def safe_log(msg: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as w:
        w.write(f"[{ts}] {msg}\n")


def load_existing(fp: Path) -> Optional[pd.DataFrame]:
    if not fp.exists():
        return None
    try:
        df = pd.read_csv(fp)
        # 보정: 필요한 컬럼만 남기되 결측 허용(추후 재계산)
        keep = [c for c in ["date", "close", "volume", "change"] if c in df.columns]
        df = df[keep].copy()
        # 타입 보정
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        df["close"] = pd.to_numeric(df.get("close"), errors="coerce")
        df["volume"] = pd.to_numeric(df.get("volume"), errors="coerce")
        # 정렬/중복제거
        df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        safe_log(f"[WARN] 기존 파일 로드 실패: {fp} -> {e}")
        return None


def compute_change_logret(df: pd.DataFrame) -> pd.DataFrame:
    """로그수익률 change = ln(close_t / close_{t-1})"""
    df = df.copy()
    df["change"] = np.log(df["close"] / df["close"].shift(1))
    return df


def fetch_krx_close_volume(ticker: str, start: date, end: date) -> pd.DataFrame:
    """
    pykrx에서 [종가, 거래량]만 추출.
    반환: date, close, volume
    """
    if start > end:
        return pd.DataFrame(columns=["date", "close", "volume"])

    df = krx.get_market_ohlcv_by_date(yyyymmdd(start), yyyymmdd(end), ticker)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "close", "volume"])

    # 인덱스: 날짜, 컬럼: 시가/고가/저가/종가/거래량/거래량/등락률 (버전별 차이 가능)
    # 안전하게 컬럼 존재 여부 확인
    close_col = "종가"
    volume_col = "거래량"
    missing = [c for c in [close_col, volume_col] if c not in df.columns]
    if missing:
        raise RuntimeError(f"[{ticker}] pykrx 결과에 컬럼 누락: {missing}")

    out = pd.DataFrame(
        {
            "date": df.index.strftime("%Y-%m-%d"),
            "close": pd.to_numeric(df[close_col], errors="coerce"),
            "volume": pd.to_numeric(df[volume_col], errors="coerce"),
        }
    )
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out = out.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
    out = out.sort_values("date").reset_index(drop=True)
    return out


def fetch_index_close_volume(index_name: str, start: date, end: date) -> pd.DataFrame:
    """
    pykrx에서 지수(예: '코스피')의 [종가, 거래량] 추출.
    반환: date, close, volume
    """
    if start > end:
        return pd.DataFrame(columns=["date", "close", "volume"])

    df = krx.get_index_ohlcv_by_date(yyyymmdd(start), yyyymmdd(end), index_name)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "close", "volume"])

    close_col = "종가"
    volume_col = "거래량"
    missing = [c for c in [close_col, volume_col] if c not in df.columns]
    if missing:
        raise RuntimeError(f"[{index_name}] pykrx 지수 결과에 컬럼 누락: {missing}")

    out = pd.DataFrame(
        {
            "date": df.index.strftime("%Y-%m-%d"),
            "close": pd.to_numeric(df[close_col], errors="coerce"),
            "volume": pd.to_numeric(df[volume_col], errors="coerce"),
        }
    )
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out = out.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
    out = out.sort_values("date").reset_index(drop=True)
    return out


def merge_and_recompute(
    old_df: Optional[pd.DataFrame], new_df: pd.DataFrame
) -> pd.DataFrame:
    """기존 + 신규 병합 후 change 전체 구간 재계산."""
    if old_df is None or old_df.empty:
        base = new_df.copy()
    else:
        base = pd.concat(
            [old_df[["date", "close", "volume"]], new_df], ignore_index=True
        )
        base = base.dropna(subset=["date"]).drop_duplicates(
            subset=["date"], keep="last"
        )
        base = base.sort_values("date").reset_index(drop=True)

    # 로그수익률 재계산
    base = compute_change_logret(base)
    return base


def determine_fetch_range(
    existing: Optional[pd.DataFrame], today: date
) -> tuple[date, date]:
    """
    내려받을 구간 결정:
    - 기존 파일 있으면: [L, today]  (L=마지막 저장일, 포함)  -> 같은 날짜를 다시 받아 덮어씀
    - 기존 파일 없으면: [today - DEFAULT_YEARS_BACK, today]
    """
    if existing is not None and not existing.empty:
        last_date: date = existing["date"].max()
        start = last_date  # 같은 날짜 포함 (덮어쓰기)
    else:
        # 최초 실행: 최근 N년
        try:
            from dateutil.relativedelta import relativedelta  # type: ignore

            start = today - relativedelta(years=DEFAULT_YEARS_BACK)
        except Exception:
            start = date(today.year - DEFAULT_YEARS_BACK, today.month, today.day)
    end = today
    return start, end


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


# ===== 메인 =====
def process_ticker(ticker: str, name: str, today: date) -> None:
    out_fp = RAW_DIR / f"{name}.csv"
    try:
        old_df = load_existing(out_fp)
        start, end = determine_fetch_range(old_df, today)
        new_df = fetch_krx_close_volume(ticker, start, end)

        if new_df.empty:
            safe_log(f"[INFO] 신규 데이터 없음: {ticker} ({start}~{end})")
            return

        final_df = merge_and_recompute(old_df, new_df)
        final_df.to_csv(out_fp, index=False)
        safe_log(
            f"[OK] {ticker} 저장: rows={len(final_df)} "
            f"(fetch {start}~{end}, new={len(new_df)}) -> {out_fp}"
        )
    except Exception as e:
        safe_log(f"[ERROR] {ticker}: {e}")


def process_index(index_name: str, output_name: str, today: date) -> None:
    """
    예) index_name='코스피', output_name='KOSPI'
    결과 저장: data/raw/KOSPI.csv
    """
    out_fp = RAW_DIR / f"{output_name}.csv"
    try:
        old_df = load_existing(out_fp)
        start, end = determine_fetch_range(old_df, today)
        new_df = fetch_index_close_volume(index_name, start, end)

        if new_df.empty:
            safe_log(f"[INFO] 신규 지수 데이터 없음: {index_name} ({start}~{end})")
            return

        final_df = merge_and_recompute(old_df, new_df)
        final_df.to_csv(out_fp, index=False)
        safe_log(
            f"[OK] {index_name} 저장: rows={len(final_df)} "
            f"(fetch {start}~{end}, new={len(new_df)}) -> {out_fp}"
        )
    except Exception as e:
        safe_log(f"[ERROR] {index_name}: {e}")


def main():
    ensure_dirs()
    if not PORTFOLIO_CSV.exists():
        raise FileNotFoundError(f"포트폴리오 파일이 없습니다: {PORTFOLIO_CSV}")

    today = now_kst_date()
    portfolio = read_portfolio(PORTFOLIO_CSV)

    # 3-1) 종목들 처리
    for _, row in portfolio.iterrows():
        code = row["code"]
        name = row["name"]
        process_ticker(code, name, today)

    # 3-2) KOSPI 지수 추가 수집
    process_index(index_name="1001", output_name="KOSPI", today=today)


if __name__ == "__main__":
    main()
