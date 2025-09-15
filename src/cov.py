"""
1년 수익률(change) 기반 종목 간 상관관계 산출

입력:
  - config/portfolio.csv            (필수 컬럼: name)
  - data/raw/{name}.csv             (컬럼: date, close, volume, change)

동작:
  1) 포트폴리오의 name 목록을 읽음
  2) 각 name의 raw CSV에서 date/ change 로드, 마지막 일자 기록
  3) 모든 파일의 '마지막 일자' 중 최솟값을 기준일(end_date)로 설정
  4) end_date로부터 365일 전(start_date) ~ end_date 구간의 change만 사용
  5) 날짜 인덱스로 병합 후 상관관계 행렬 계산 (pairwise, NaN 무시)
  6) data/processed/corr_1y_{end_date}.csv 저장

출력:
  - data/processed/corr_1y_YYYY-MM-DD.csv
"""

from __future__ import annotations

from pathlib import Path
from datetime import timedelta
from datetime import date as _date
import sys
import pandas as pd

# ===== 경로/설정 =====
PORTFOLIO_CSV = Path("config/portfolio.csv")
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RESULT_DIR = Path("data/result")
WINDOW_DAYS = 365  # 1년 창(일수)
MIN_OVERLAP = None  # 상관계수 유효판정 위한 최소 겹침 일수 (None이면 기본 동작)
ROUND_DECIMALS = 4  # 결과 반올림 자릿수


def _read_portfolio_names(fp: Path) -> list[str]:
    if not fp.exists():
        print(f"[오류] 포트폴리오 파일이 없습니다: {fp}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(fp)
    if "name" not in df.columns:
        print(f"[오류] {fp}에 'name' 컬럼이 없습니다.", file=sys.stderr)
        sys.exit(1)
    names = [str(x).strip() for x in df["name"].dropna().tolist()]
    # 중복 제거(원래 순서 유지)
    seen, uniq = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq


def _load_change_series(name: str):
    """
    data/raw/{name}.csv를 읽어 (마지막 일자, change 시리즈) 반환
    - 시리즈 index: DatetimeIndex(date), name: 종목명
    - change는 float로 강제 변환 후 NaN 제거
    """
    fp = RAW_DIR / f"{name}.csv"
    if not fp.exists():
        print(f"[경고] 파일 없음: {fp} -> 건너뜀")
        return None

    try:
        df = pd.read_csv(fp, parse_dates=["date"])
    except Exception as e:
        print(f"[경고] 파일 읽기 실패: {fp} ({e}) -> 건너뜀")
        return None

    # 필수 컬럼 체크
    need = {"date", "change"}
    if not need.issubset(df.columns.str.lower()):
        # 소문자 비교 후 실제 컬럼명 매핑
        cols_lower = {c.lower(): c for c in df.columns}
        if not need.issubset(cols_lower.keys()):
            print(f"[경고] 필수 컬럼 누락(date/change): {fp} -> 건너뜀")
            return None
        # 필요시 원래 컬럼명으로 접근
        date_col = cols_lower["date"]
        change_col = cols_lower["change"]
    else:
        date_col = "date"
        change_col = "change"

    # 정리
    s = (
        df[[date_col, change_col]]
        .rename(columns={date_col: "date", change_col: "change"})
        .dropna(subset=["date"])
    )
    # change 숫자화
    s["change"] = pd.to_numeric(s["change"], errors="coerce")
    s = s.dropna(subset=["change"]).sort_values("date")

    if s.empty:
        print(f"[경고] 유효 데이터 없음(빈 change): {fp} -> 건너뜀")
        return None

    s = s.set_index("date")["change"].copy()
    s.name = name
    last_date = s.index.max()
    return last_date, s


def main():
    names = _read_portfolio_names(PORTFOLIO_CSV)
    if not names:
        print("[오류] 포트폴리오에 name 항목이 비어 있습니다.", file=sys.stderr)
        sys.exit(1)

    loaded = []
    last_dates = []
    for nm in names:
        res = _load_change_series(nm)
        if res is None:
            continue
        last_date, series = res
        loaded.append(series)
        last_dates.append(last_date)

    if not loaded:
        print("[오류] 로드된 시리즈가 없습니다(모든 파일 누락/에러).", file=sys.stderr)
        sys.exit(1)

    # 공통 기준일: 각 파일의 '마지막 일자' 중 최솟값
    end_date = min(last_dates)
    start_date = end_date - timedelta(days=WINDOW_DAYS)

    # 1년 구간 슬라이싱 + 병합
    trimmed = []
    for s in loaded:
        s_win = s.loc[(s.index >= start_date) & (s.index <= end_date)]
        if s_win.empty:
            # 해당 종목은 창 내 데이터 없음 -> 일단 포함하지 않음
            print(f"[정보] 창 내 데이터 없음: {s.name} (제외)")
            continue
        trimmed.append(s_win)

    if len(trimmed) < 2:
        print(
            "[오류] 상관계수를 계산할 최소 2개 종목의 유효 구간 데이터가 없습니다.",
            file=sys.stderr,
        )
        sys.exit(1)

    mat = pd.concat(trimmed, axis=1)  # 날짜 기준 outer-join
    # 상관관계 (pairwise, NaN 무시). min_periods로 겹침 최소일수 설정 가능
    corr = mat.corr(min_periods=MIN_OVERLAP).round(ROUND_DECIMALS)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_fp = PROCESSED_DIR / f"corr_1y_{end_date.date()}.csv"
    corr.to_csv(out_fp, encoding="utf-8-sig")

    print(f"[완료] {len(trimmed)}개 종목, 창: {start_date.date()} ~ {end_date.date()}")
    print(f"[저장] {out_fp}")

    asof = end_date.date() if "end_date" in locals() else _date.today()

    out_fp = RESULT_DIR / f"portfolio_status_{asof.isoformat()}.md"

    def _fmt_corr(v, nd=ROUND_DECIMALS if "ROUND_DECIMALS" in locals() else 4):
        return "" if pd.isna(v) else f"{v:.{nd}f}"

    # Markdown 섹션 + 테이블(커스텀 형식)
    lines = []
    lines.append("## 종목 간 상관관계 (1년)\n")
    lines.append(
        f"- 기준일: {end_date.date().isoformat()}  | 기간: {start_date.date().isoformat()} ~ {end_date.date().isoformat()}\n\n"
    )

    cols = list(corr.columns)
    # 헤더
    lines.append("| name | " + " | ".join(cols) + " |")
    # 정렬 라인: name은 좌, 나머지는 우정렬
    lines.append("|---|" + "|".join(["---:" for _ in cols]) + "|")

    # 본문 행
    for idx in corr.index:
        row_vals = [_fmt_corr(corr.loc[idx, c]) for c in cols]
        lines.append("| " + str(idx) + " | " + " | ".join(row_vals) + " |")

    # 테이블 종료 공백 줄
    lines.append("")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_fp, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
