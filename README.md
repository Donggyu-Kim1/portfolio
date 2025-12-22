## 1. 프로젝트 개요
국내 KRX 포트폴리오에 대해 **가격 수집 → 수익률·베타·변동성 계산 → 상관/자기상관 분석 → 10일 99% VaR 및 백테스트**를 자동화해 리스크를 빠르게 모니터링하는 파이프라인입니다.

## 2. 핵심 기능
- **입력/설정**: `config/portfolio.csv`(필수 컬럼: `code`, `name`, `shares`, `price`)로 종목/보유정보를 정의하고, 벤치마크 `data/raw/KOSPI.csv`를 함께 준비합니다. 존재하지 않거나 필수 컬럼이 없으면 파이프라인이 중단됩니다.
- **`src/price.py`**: `portfolio.csv`를 읽어 KRX 종목/지수의 일별 종가·거래량을 증분 수집하고 로그수익률(`change`)을 재계산해 `data/raw/*.csv`를 유지합니다.
- **`src/return.py`**: 포트폴리오 누적·일일 수익률과 KOSPI 대비 초과수익률을 계산해 Markdown 보고서를 생성합니다.
- **`src/beta.py`**: 최근 1년 일간 로그수익률로 종목별/포트폴리오 Beta를 산출하고 보고서에 섹션을 추가합니다.
- **`src/ewma.py`**: RiskMetrics λ=0.94 EWMA로 1년 일간 변동성을 추정해 CSV 및 보고서 섹션을 생성합니다.
- **`src/cov.py`**: 최근 1년 수익률 기반 종목 간 상관행렬을 계산해 CSV와 보고서 섹션을 기록합니다.
- **`src/self_cov.py`**: 최근 1년 자기상관(1~9 lag)과 10일 스케일 계수를 계산해 ACF CSV를 생성합니다.
- **`src/var.py`**: EWMA·ACF·상관행렬을 통합해 개별/포트폴리오 10일 99% VaR, Component VaR, 15% 임계 초과 여부를 산출하고 CSV/Markdown을 작성합니다.
- **`src/backtesting.py`**: 5년 롤링 창으로 10일 99% VaR 백테스트(Christoffersen/Kupiec 테스트 포함)를 실행해 CSV/MD 요약을 생성합니다.
- **`main.py`**: 모든 스크립트를 순차 실행하는 파이프라인 러너(`python main.py [--keep-going] [-- ...]`), 콘솔 및 파일 로그를 관리합니다.

## 3. 기술 스택
- **Python 3 + 표준 라이브러리**: `argparse`, `subprocess`, `pathlib`, `datetime` 등으로 파이프라인 제어 및 로깅.
- **데이터/과학 라이브러리**: `pandas`, `numpy`, `math`, `dataclasses`로 시계열 처리와 통계 계산.
- **금융 데이터 수집**: `pykrx`로 KRX 가격/거래량 다운로드.
- **보조 도구**: `dateutil.relativedelta`(선택적)로 1년 창 계산, `ZoneInfo`로 KST 기준 날짜 처리.

## 4. 아키텍처 / 데이터 흐름
1. **입력 준비**: `config/portfolio.csv`에 종목 코드/이름/수량/매수가를 정의하고, 벤치마크 `data/raw/KOSPI.csv`를 준비합니다.
2. **가격 수집** (`price.py`): 종목·KOSPI의 일별 종가/거래량을 내려받아 `data/raw/*.csv`를 갱신하고 `change`를 재계산합니다.
3. **초기 보고서 생성** (`return.py`): 누적/일일 수익률과 초과수익률을 계산해 `portfolio_status_{YYYY-MM-DD}.md`를 생성합니다.
4. **보조 지표 추가**:  
   - Beta 섹션 append (`beta.py`).  
   - EWMA 변동성 섹션 append (`ewma.py`).  
   - 1년 상관행렬 섹션 append (`cov.py`).  
   - ACF/스케일 CSV 생성 (`self_cov.py`).  
5. **포트폴리오 VaR 산출** (`var.py`): EWMA/ACF/상관행렬을 합쳐 개별·포트폴리오 VaR, Component VaR, 15% 임계 초과 여부를 계산하고 MD/CSV로 기록합니다.
6. **백테스트** (`backtesting.py`): 5년 롤링으로 VaR 적중률·독립성 검증 및 Tail Calibration 요약을 MD/CSV로 생성합니다.
7. **파이프라인 실행** (`main.py`): 전체 단계를 순차 실행하며 실패 시 중단/계속 옵션과 로그(`data/_logs/pipeline_*.log`)를 관리합니다.

## 5. 산출물 및 보고서 구성
- **원천 시계열**: `data/raw/{name}.csv`, `data/raw/KOSPI.csv` (date, close, volume, change).
- **가공 데이터**:  
  - EWMA 변동성 CSV: `data/processed/ewma_1y_lambda0_94_{asof}.csv`.  
  - ACF/스케일 CSV: `data/processed/acf_1y_lag1_9_{asof}.csv`.  
  - 상관행렬 CSV: `data/processed/corr_1y_{end_date}.csv`.
- **보고서 파일(주요 구조)**: `data/result/portfolio_status_{YYYY-MM-DD}.md`  
  - 머리말 및 요약(누적/일일 수익률, 초과수익률) + 개별 종목 표: `return.py`에서 생성.  
  - **Beta 섹션**: “## Beta (1Y, 일간 로그수익률)” 표와 요약을 append.  
  - **EWMA Vol 섹션**: “## EWMA Vol (1Y, λ=0.94, Daily)” 표를 append.  
  - **상관행렬 섹션**: “## 종목 간 상관관계 (1년)” 표를 append.  
  - **포트폴리오 VaR 섹션**: 10일 99% 포트폴리오 VaR 요약, 임계(15%) 여부, Top 10 Component VaR 표를 append.
- **추가 리포트**:  
  - VaR 백테스트: `data/result/var_backtest_portfolio_{start}_{end}.csv/.md`.  
  - VaR 개별/포트폴리오 CSV: `data/result/var_individual_{asof}.csv`, `data/result/var_portfolio_{asof}.csv`.
- **로그**: 파이프라인 및 가격 수집 로그 `data/_logs/`.
