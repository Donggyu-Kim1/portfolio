from datetime import date, timedelta
from pathlib import Path
import os
import pandas as pd
import numpy as np
from pykrx import stock

# 폰트/마이너스 설정 (pyplot import 전에)
import matplotlib
from matplotlib import font_manager as fm

CANDIDATES = [
    "Noto Sans CJK KR",  # Ubuntu noto-cjk
    "NanumGothic",  # Ubuntu fonts-nanum
    "AppleGothic",  # macOS
    "Malgun Gothic",  # Windows
]

available = {f.name for f in fm.fontManager.ttflist}
picked = None
for name in CANDIDATES:
    if name in available:
        picked = name
        break

if picked is None:
    print("⚠️  CJK 폰트를 찾지 못했습니다. 한글 표시가 깨질 수 있어요.")
else:
    matplotlib.rcParams["font.family"] = picked

matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt

PORTFOLIO_FILE = Path("config/portfolio.csv")


def load_portfolio_csv(path=PORTFOLIO_FILE):
    """
    config/portfolio.csv 에서 code,name,shares,price를 읽어 정합성 검증 후
    (동일 code 중복 시) shares 합산, price는 주수 가중 평균으로 병합해 반환.
    반환: df(코드/이름/수량/매입가), tickers{label->code}, shares{label->shares}, buy_price(Series)
    label은 열 이름에 쓰일 컬럼(이름 있으면 이름, 없으면 코드)
    """
    df = pd.read_csv(path, dtype={"code": str})
    # 정리
    df["code"] = df["code"].str.strip().str.zfill(6)
    if "name" not in df.columns:
        df["name"] = ""
    df["name"] = df["name"].fillna("").astype(str).str.strip()
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # 검증
    if df["code"].isna().any():
        raise ValueError("code에 결측치가 있습니다.")
    if df["shares"].isna().any():
        bad = df[df["shares"].isna()]
        raise ValueError(f"shares가 숫자가 아닙니다: {bad}")
    if (df["shares"] < 0).any():
        bad = df[df["shares"] < 0]
        raise ValueError(f"음수 shares 존재: {bad}")
    if df["price"].isna().any() or (df["price"] <= 0).any():
        bad = df[df["price"].isna() | (df["price"] <= 0)]
        raise ValueError(f"price(매입단가)가 0 이하 또는 결측: {bad}")

    # 동일 code 중복 병합: shares 합, price는 주수 가중 평균
    if df.duplicated("code").any():
        df = (
            df.assign(weight=lambda x: x["shares"])
            .groupby("code", as_index=False)
            .apply(
                lambda g: pd.Series(
                    {
                        "name": (
                            g.loc[g["name"] != "", "name"].iloc[0]
                            if (g["name"] != "").any()
                            else ""
                        ),  # 첫 유효 이름
                        "shares": g["shares"].sum(),
                        "price": (g["price"] * g["shares"]).sum() / g["shares"].sum(),
                    }
                )
            )
        )

    # 이름 비어있으면 KRX에서 채우기 (가능한 경우)
    mask_empty = df["name"] == ""
    if mask_empty.any():
        for i in df.index[df["name"] == ""]:
            try:
                df.at[i, "name"] = stock.get_market_ticker_name(df.at[i, "code"])
            except Exception:
                df.at[i, "name"] = df.at[i, "code"]  # 최후: 코드로 대체

    # label = 표/그래프 축에 쓸 이름 (가급적 종목명, 없으면 코드)
    df["label"] = df.apply(lambda r: r["name"] if r["name"] else r["code"], axis=1)

    # 딕셔너리/시리즈 생성
    tickers = dict(zip(df["label"], df["code"]))
    shares = dict(zip(df["label"], df["shares"].astype(int)))
    buy_price = pd.Series(df["price"].values, index=df["label"], name="buy_price")

    return df[["code", "name", "shares", "price", "label"]], tickers, shares, buy_price


# ---------- 0) 날짜/폴더 세팅 ----------
today = date.today()
run_str = today.strftime("%Y%m%d")

ROOT = os.path.join("data", run_str)
DIRS = {
    "prices": os.path.join(ROOT, "prices"),
    "returns": os.path.join(ROOT, "returns"),
    "correlation": os.path.join(ROOT, "correlation"),
    "risk_summary": os.path.join(ROOT, "risk_summary"),
    "visuals": os.path.join(ROOT, "visuals"),
}
for p in DIRS.values():
    os.makedirs(p, exist_ok=True)

# ---------- 1) 기본 설정 ----------
portfolio_df, tickers, shares, buy_price = load_portfolio_csv()

end = today
start = end - timedelta(days=365 * 3)
to_krx = lambda d: d.strftime("%Y%m%d")
start_str, end_str = to_krx(start), to_krx(end)

# ---------- 2) 10종목 일별 종가 (수정주가 반영) ----------
panel = {}
for name, code in tickers.items():
    df = stock.get_market_ohlcv(start_str, end_str, code)
    panel[name] = df["종가"].rename(name)

prices = pd.concat(panel.values(), axis=1)
prices.index.name = "날짜"

# ---------- 3) KOSPI 지수(가격지수) ----------
kospi = stock.get_index_ohlcv_by_date(start_str, end_str, "1001")["종가"].rename(
    "KOSPI"
)

# ---------- 4) 날짜 정합 ----------
aligned = prices.join(kospi, how="inner").dropna(how="any")

# ---------- 5) 저장(가격) ----------
prices_path = os.path.join(DIRS["prices"], f"krx_10stocks_close_{run_str}.csv")
aligned.to_csv(prices_path, encoding="utf-8-sig")
print(aligned.shape, "Saved:", prices_path)

# ---------- 6) 로그수익률 ----------
logret = np.log(aligned / aligned.shift(1)).dropna()

# ---------- 7) 시장/종목 분리 ----------
mkt = logret["KOSPI"]
stocks = logret.drop(columns=["KOSPI"])

# ---------- 8) 연율화 변동성 ----------
vol_annual = stocks.std() * np.sqrt(252)
vol_annual.name = "vol_annual"

# ---------- 9) 베타 ----------
mkt_var = mkt.var()
beta = stocks.apply(lambda s: s.cov(mkt) / mkt_var)
beta.name = "beta"


# ---------- 10) 최대낙폭 ----------
def max_drawdown(price_series):
    cummax = price_series.cummax()
    drawdown = price_series / cummax - 1.0
    return drawdown.min()


mdd = aligned.drop(columns=["KOSPI"]).apply(max_drawdown)
mdd.name = "max_drawdown"

# ---------- 11) 상관행렬 ----------
corr_mat = stocks.corr()

# ---------- 12) 요약/저장 ----------
summary = pd.concat([vol_annual, beta, mdd], axis=1).sort_values(
    "vol_annual", ascending=False
)

ret_path = os.path.join(DIRS["returns"], f"krx_10stocks_logret_{run_str}.csv")
summ_path = os.path.join(
    DIRS["risk_summary"], f"krx_10stocks_risk_summary_{run_str}.csv"
)
corr_path = os.path.join(DIRS["correlation"], f"krx_10stocks_corr_matrix_{run_str}.csv")

logret.to_csv(ret_path, encoding="utf-8-sig")
summary.to_csv(summ_path, encoding="utf-8-sig")
corr_mat.to_csv(corr_path, encoding="utf-8-sig")

print("Rows (returns):", logret.shape[0])
print("Saved:", ret_path)
print("Saved:", summ_path)
print("Saved:", corr_path)

# ---------- 13) 시각화 저장 ----------
# 변동성 히트맵
vol = summary["vol_annual"].copy()
vol_pct = (vol * 100).round(2)

fig1, ax1 = plt.subplots(figsize=(max(8, len(vol) * 0.8), 2.2))
data_vol = vol.values.reshape(1, -1)
im1 = ax1.imshow(data_vol, aspect="auto")  # 색상 지정 X (기본)

ax1.set_yticks([0])
ax1.set_yticklabels(["Vol (annual, %)"])
ax1.set_xticks(np.arange(len(vol.index)))
ax1.set_xticklabels(vol.index, rotation=45, ha="right")

for j, v in enumerate(vol_pct.values):
    ax1.text(j, 0, f"{v}%", va="center", ha="center", fontsize=9)

cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label("Annualized Volatility")

plt.tight_layout()
vol_fig_path = os.path.join(DIRS["visuals"], f"heatmap_volatility_{run_str}.png")
plt.savefig(vol_fig_path, dpi=150)
plt.close(fig1)

# 상관 히트맵
fig2, ax2 = plt.subplots(
    figsize=(max(8, len(corr_mat) * 0.8), max(6, len(corr_mat) * 0.6))
)
im2 = ax2.imshow(corr_mat.values, aspect="equal", vmin=-1, vmax=1)

ax2.set_xticks(np.arange(len(corr_mat.columns)))
ax2.set_yticks(np.arange(len(corr_mat.index)))
ax2.set_xticklabels(corr_mat.columns, rotation=45, ha="right")
ax2.set_yticklabels(corr_mat.index)

for i in range(corr_mat.shape[0]):
    for j in range(corr_mat.shape[1]):
        ax2.text(
            j, i, f"{corr_mat.values[i,j]:.2f}", va="center", ha="center", fontsize=8
        )

cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label("Correlation")

plt.title("Stocks Correlation Matrix")
plt.tight_layout()
corr_fig_path = os.path.join(DIRS["visuals"], f"heatmap_correlation_{run_str}.png")
plt.savefig(corr_fig_path, dpi=150)
plt.close(fig2)

print("Saved visuals:", vol_fig_path, " / ", corr_fig_path)

# ---------- 14) 포트폴리오 (보유 수량 반영) ----------
# (1) 현재가 & 평가금액 & 비중
labels = list(shares.keys())  # 라벨(=CSV의 name 또는 code) 순서
latest_prices = aligned.iloc[-1][labels]

values = latest_prices * pd.Series(shares)
total_value = float(values.sum())
weights = (values / total_value).sort_values(ascending=False)  # 비중 내림차순
weights.name = "weight"

# === [추가] 매입단가 기반 원금/손익 ===
buy = buy_price.reindex(labels)  # 매입단가(원/주)
costs = buy * pd.Series(shares)  # 종목별 원금
pnl = values - costs  # 평가손익(원)
pnl_pct = (pnl / costs.replace(0, np.nan)) * 100  # 손익률(%)

# 저장: 종목별 평가금액/비중
port_comp = pd.concat(
    [
        latest_prices.reindex(weights.index).rename("last_price"),
        buy.reindex(weights.index).rename("buy_price"),
        pd.Series(shares).reindex(weights.index).rename("shares"),
        costs.reindex(weights.index).rename("cost"),
        values.reindex(weights.index).rename("value"),
        pnl.reindex(weights.index).rename("pnl"),
        pnl_pct.reindex(weights.index).rename("pnl_pct"),
        weights,
    ],
    axis=1,
)
comp_path = os.path.join(DIRS["risk_summary"], f"portfolio_composition_{run_str}.csv")
port_comp.to_csv(comp_path, encoding="utf-8-sig")
print("Saved:", comp_path)

# (2) 공분산 행렬(연율), 포트폴리오 변동성/분산
cov_matrix = stocks.cov() * 252
w = weights.reindex(cov_matrix.columns).fillna(0).values  # 종목 순서 정합
port_var = float(np.dot(w, np.dot(cov_matrix.values, w)))
port_vol_annual = float(np.sqrt(port_var))

# (3) 포트폴리오 베타 (가중합)
beta_vec = beta.reindex(weights.index).fillna(0)
port_beta = float((weights * beta_vec).sum())

# (4) 1일 VaR (정규 가정, 95%/99%)
z_95, z_99 = 1.65, 2.33
sigma_day = port_vol_annual / np.sqrt(252)
VaR95 = z_95 * sigma_day * total_value
VaR99 = z_99 * sigma_day * total_value

# (5) 리스크 기여도 (Component / Marginal)
# marginal = Σ w  (벡터), component = w_i * marginal_i
marginal = cov_matrix.values.dot(w)  # 각 종목의 한계 위험
component = w * marginal  # 분산 기여도
risk_contrib_pct = pd.Series(component / port_var, index=cov_matrix.columns) * 100.0
risk_contrib_pct = risk_contrib_pct.reindex(weights.index)  # 비중 순서에 맞춤

# 저장: 리스크 기여도
risk_contrib_df = pd.DataFrame(
    {
        "weight": weights,
        "beta": beta_vec.reindex(weights.index),
        "risk_contrib_pct": risk_contrib_pct,
    }
)
risk_contrib_path = os.path.join(
    DIRS["risk_summary"], f"portfolio_risk_contrib_{run_str}.csv"
)
risk_contrib_df.to_csv(risk_contrib_path, encoding="utf-8-sig")

# ---------- 15) 포트폴리오 요약 출력 ----------
summary_txt = (
    f"Total Value: {total_value:,.0f} KRW\n"
    f"Portfolio Vol (annual): {port_vol_annual*100:.2f}%\n"
    f"Portfolio Beta vs KOSPI: {port_beta:.3f}\n"
    f"1-day VaR 95%: {VaR95:,.0f} KRW | 99%: {VaR99:,.0f} KRW\n"
)
print(summary_txt)

# ---------- 16) 그래프 저장 (matplotlib만, 색상 지정 없이, 단일 플롯씩) ----------
# (A) 비중 막대그래프
figA, axA = plt.subplots(figsize=(max(8, len(weights) * 0.6), 4))
axA.bar(weights.index, weights.values)
axA.set_title("Portfolio Weights")
axA.set_ylabel("Weight")
axA.set_ylim(0, max(0.25, weights.max() * 1.2))
axA.set_xticklabels(weights.index, rotation=45, ha="right")
for i, v in enumerate(weights.values):
    axA.text(i, v, f"{v*100:.1f}%", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
weights_fig = os.path.join(DIRS["visuals"], f"portfolio_weights_{run_str}.png")
plt.savefig(weights_fig, dpi=150)
plt.close(figA)

# (B) 리스크 기여도 막대그래프 (% of portfolio variance)
figB, axB = plt.subplots(figsize=(max(8, len(risk_contrib_pct) * 0.6), 4))
axB.bar(risk_contrib_pct.index, risk_contrib_pct.values)
axB.set_title("Risk Contribution (% of Portfolio Variance)")
axB.set_ylabel("%")
axB.set_ylim(0, max(20, risk_contrib_pct.max() * 1.2))
axB.set_xticklabels(risk_contrib_pct.index, rotation=45, ha="right")
for i, v in enumerate(risk_contrib_pct.values):
    axB.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
riskc_fig = os.path.join(DIRS["visuals"], f"portfolio_risk_contribution_{run_str}.png")
plt.savefig(riskc_fig, dpi=150)
plt.close(figB)

# (C) 종목 베타 막대그래프
beta_ord = beta_vec.reindex(weights.index)
figC, axC = plt.subplots(figsize=(max(8, len(beta_ord) * 0.6), 4))
axC.bar(beta_ord.index, beta_ord.values)
axC.axhline(1.0)  # 시장 베타 기준선
axC.set_title("Stock Betas (vs KOSPI)")
axC.set_ylabel("Beta")
axC.set_xticklabels(beta_ord.index, rotation=45, ha="right")
for i, v in enumerate(beta_ord.values):
    axC.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
beta_fig = os.path.join(DIRS["visuals"], f"stock_betas_{run_str}.png")
plt.savefig(beta_fig, dpi=150)
plt.close(figC)

print("Saved visuals:", weights_fig, " / ", riskc_fig, " / ", beta_fig)

# (D) 종목별 손익 그래프
figD, axD = plt.subplots(figsize=(max(8, len(pnl) * 0.6), 4))
pnl_ord = pnl.reindex(weights.index)  # 비중 순서로 정렬
axD.bar(pnl_ord.index, pnl_ord.values)
axD.set_title("Unrealized P&L by Stock (KRW)")
axD.set_ylabel("KRW")
axD.set_xticklabels(pnl_ord.index, rotation=45, ha="right")
for i, v in enumerate(pnl_ord.values):
    axD.text(
        i, v, f"{v:,.0f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8
    )
plt.tight_layout()
pnl_fig = os.path.join(DIRS["visuals"], f"pnl_by_stock_{run_str}.png")
plt.savefig(pnl_fig, dpi=150)
plt.close(figD)
print("Saved visual:", pnl_fig)

# ---------- 17) 포트폴리오 지표 요약 CSV ----------
total_cost = float(costs.sum())
total_pnl = float((values - costs).sum())
total_pnl_pct = (total_pnl / total_cost) if total_cost > 0 else float("nan")

port_metrics = pd.DataFrame(
    {
        "metric": [
            "total_cost_krw",
            "total_value_krw",
            "total_pnl_krw",
            "total_pnl_pct",
            "vol_annual",
            "beta",
            "VaR95_1d_krw",
            "VaR99_1d_krw",
        ],
        "value": [
            total_cost,
            total_value,
            total_pnl,
            total_pnl_pct,
            port_vol_annual,
            port_beta,
            VaR95,
            VaR99,
        ],
    }
)
metrics_path = os.path.join(DIRS["risk_summary"], f"portfolio_metrics_{run_str}.csv")
port_metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
print("Saved:", metrics_path)

# 리포트 폴더
REPORT_DIR = os.path.join(ROOT, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# 1) 집중도/상관/요약 수치 계산
# 비중 HHI (집중도 지수)
hhi = float((weights**2).sum())

# 평균 상관계수 (대각 제외)
corr_vals = corr_mat.values
mask = ~np.eye(corr_vals.shape[0], dtype=bool)
avg_corr = float(corr_vals[mask].mean())

# 상위 리스크 기여도 & 비중
risk_top1_name = risk_contrib_df.sort_values("risk_contrib_pct", ascending=False).index[
    0
]
risk_top1_pct = float(risk_contrib_df.loc[risk_top1_name, "risk_contrib_pct"])
risk_top3_pct = float(
    risk_contrib_df.sort_values("risk_contrib_pct", ascending=False)["risk_contrib_pct"]
    .head(3)
    .sum()
)

w_top1_name = weights.index[0]
w_top1 = float(weights.iloc[0])
w_top3 = float(weights.head(3).sum())

# 2) 포트폴리오 과거 평가액/수익률 & MDD
#    - 현재 보유 수량 고정 가정으로 과거 평가액 계산
stock_cols = list(weights.index)
portfolio_value_series = (aligned[stock_cols] * pd.Series(shares)[stock_cols]).sum(
    axis=1
)
portfolio_value_series.name = "portfolio_value"
# MDD
cummax = portfolio_value_series.cummax()
port_drawdown = portfolio_value_series / cummax - 1.0
port_mdd = float(port_drawdown.min())

# 60일 롤링 변동성(연율화)
port_logret = np.log(portfolio_value_series / portfolio_value_series.shift(1)).dropna()
rolling_vol_60d = port_logret.rolling(60).std() * np.sqrt(252)
vol_60d_latest = (
    float(rolling_vol_60d.dropna().iloc[-1])
    if len(rolling_vol_60d.dropna())
    else float("nan")
)

# === 투자 성과 (일간) & 종목별 현재 수익률 표 ===
# 포트폴리오/코스피 '일간 수익률' (단순 수익률)
port_daily_ret_series = portfolio_value_series.pct_change().dropna()
kospi_daily_ret_series = aligned["KOSPI"].pct_change().dropna()
port_daily_ret = (
    float(port_daily_ret_series.iloc[-1])
    if len(port_daily_ret_series)
    else float("nan")
)
kospi_daily_ret = (
    float(kospi_daily_ret_series.iloc[-1])
    if len(kospi_daily_ret_series)
    else float("nan")
)
excess_daily_ret = (
    port_daily_ret - kospi_daily_ret
    if np.isfinite(port_daily_ret) and np.isfinite(kospi_daily_ret)
    else float("nan")
)

# 종목별 현재 수익률(매입가 대비) 테이블 (가독성을 위해 비중 순서로)
order = list(weights.index)
rows = [
    "| 종목 | 수량 | 매입가 | 현재가 | 손익률 | 손익(원) |",
    "|---|---:|---:|---:|---:|---:|",
]
for lbl in order:
    qty = int(pd.Series(shares).get(lbl, 0))
    bp = float(buy.get(lbl, np.nan))  # 매입가
    lp = float(latest_prices.get(lbl, np.nan))  # 현재가
    retpct = (lp / bp - 1) * 100 if bp > 0 and np.isfinite(lp) else float("nan")
    pnl_val = (lp - bp) * qty if np.isfinite(lp) and np.isfinite(bp) else float("nan")
    rows.append(
        f"| {lbl} | {qty:,} | {bp:,.0f} | {lp:,.0f} | {retpct:.2f}% | {pnl_val:,.0f} |"
    )

perf_table_md = "\n".join(rows)

# 리포트용 섹션 문자열
perf_md = (
    "## 투자 성과\n"
    f"- 일간 수익률: 포트 {(port_daily_ret*100):.2f}% · KOSPI {(kospi_daily_ret*100):.2f}% · 초과수익 {(excess_daily_ret*100):.2f}%\n\n"
    "### 종목별 현재 수익률 (매입가 대비)\n"
    f"{perf_table_md}\n"
)

# 3) 규칙 기반 메시지 생성 (임계값은 필요 시 조정)
messages = []

# (A) 비중/리스크 집중 경고
if w_top1 > 0.20:
    messages.append(
        f"- 비중 집중: 상위 1개 종목 비중이 {w_top1*100:.1f}%로 20% 초과. 단일 종목 리스크가 큽니다."
    )
if w_top3 > 0.60:
    messages.append(
        f"- 비중 집중: 상위 3개 종목 합계 비중 {w_top3*100:.1f}% (60% 초과). 분산 저하 우려."
    )
if risk_top1_pct > 25:
    messages.append(
        f"- 리스크 집중: '{risk_top1_name}'의 리스크 기여도가 {risk_top1_pct:.1f}%로 높음. 변동성 과대 기여."
    )
if risk_top3_pct > 60:
    messages.append(
        f"- 리스크 집중: 상위 3개 종목 리스크 기여 합계 {risk_top3_pct:.1f}% (60% 초과)."
    )

# (B) 상관/분산효과
if avg_corr > 0.6:
    messages.append(
        f"- 상관 고도화: 평균 상관계수 {avg_corr:.2f} (0.60↑). 시장/업종 동조화로 분산효과가 낮습니다."
    )
elif avg_corr < 0.2:
    messages.append(
        f"- 낮은 상관: 평균 상관계수 {avg_corr:.2f}. 분산효과가 비교적 좋습니다."
    )

# (C) 시장 노출(베타)
if port_beta > 1.1:
    messages.append(
        f"- 공격적 노출: 포트폴리오 베타 {port_beta:.2f} (1.10↑). 시장 하락 시 손실 확대 가능."
    )
elif port_beta < 0.9:
    messages.append(
        f"- 방어적 노출: 포트폴리오 베타 {port_beta:.2f} (0.90↓). 랠리 구간에선 초과수익 어려울 수 있음."
    )

# (D) 리스크 레벨 & 손실허용치
messages.append(
    f"- 변동성 수준: 연 변동성 {port_vol_annual*100:.2f}% (최근 60일 추정 {vol_60d_latest*100:.2f}% 참고)."
)
messages.append(
    f"- 손실한도(VaR): 1일 VaR95 {VaR95:,.0f}원 / 99% {VaR99:,.0f}원 (총 평가액 {total_value:,.0f}원 기준)."
)
messages.append(f"- 역사적 낙폭: 누적 기준 최대낙폭(MDD) {port_mdd*100:.1f}%.")

# 4) 액션 제안(간단 규칙)
actions = []
# 리스크 기여도 상한 20% 가정 → 넘는 종목 감축 제안
THRESH_RC = 20.0
over_rc = risk_contrib_df[risk_contrib_df["risk_contrib_pct"] > THRESH_RC]
if not over_rc.empty:
    names = ", ".join(
        [f"{idx}({rc:.1f}%)" for idx, rc in over_rc["risk_contrib_pct"].items()]
    )
    actions.append(
        f"- 리스크 상한 적용: 리스크 기여도 {THRESH_RC:.0f}% 초과 종목 감축 고려 → {names}."
    )
# 변동성 타깃팅 예: 목표 연변동성 12%
TARGET_VOL = 0.12
if port_vol_annual > TARGET_VOL:
    scale = TARGET_VOL / port_vol_annual
    actions.append(
        f"- 변동성 타깃팅: 목표 12% 기준 익스포저 ×{scale:.2f} 축소(현금 비중/헤지 활용)."
    )
# 베타 중립화(선택): 베타 1.0 근접화
if port_beta > 1.05:
    actions.append("- 베타 중립화: 고베타 종목 비중↓, 저베타 종목↑로 시장 민감도 완화.")
elif port_beta < 0.95:
    actions.append(
        "- 베타 조정: 저베타 비중↓ 혹은 베타 높은 종목 일부↑로 시장 민감도 상향."
    )
# 상관 완화: 동일 업종/높은 상관 쌍 비중 축소
if avg_corr > 0.6:
    actions.append(
        "- 상관 완화: 동일 업종/고상관 쌍 중 탑다운 선택해 일부 축소(대체 섹터/현금)."
    )

# 5) 리포트 텍스트 구성
header = (
    f"# 포트폴리오 리스크 인사이트 ({run_str})\n"
    f"- 총 평가액: {total_value:,.0f}원\n"
    f"- 연 변동성: {port_vol_annual*100:.2f}% | 베타: {port_beta:.2f}\n"
    f"- 1일 VaR95/99: {VaR95:,.0f} / {VaR99:,.0f} 원\n"
    f"- 포트폴리오 MDD: {port_mdd*100:.1f}%\n"
    f"- 평균 상관계수: {avg_corr:.2f} | HHI(비중 집중): {hhi:.3f}\n"
    f"- 상위 비중: 1개 {w_top1*100:.1f}% ( {w_top1_name} ), 상위3개 {w_top3*100:.1f}%\n"
    f"- 상위 리스크 기여: 1개 {risk_top1_pct:.1f}% ( {risk_top1_name} ), 상위3개 {risk_top3_pct:.1f}%\n"
)

insights = "## 인사이트\n" + ("\n".join(messages) if messages else "- 특이사항 없음.")
reco = "## 액션 제안\n" + (
    "\n".join(actions) if actions else "- 즉시 조치 필요 없음. 정기 모니터링 유지."
)

report_md = header + "\n" + perf_md + "\n" + insights + "\n\n" + reco + "\n"
report_txt = report_md.replace("#", "").replace("*", "")

# 6) 저장
md_path = os.path.join(REPORT_DIR, f"portfolio_insight_{run_str}.md")
txt_path = os.path.join(REPORT_DIR, f"portfolio_insight_{run_str}.txt")
with open(md_path, "w", encoding="utf-8") as f:
    f.write(report_md)
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(report_txt)

print("Saved report:", md_path, " / ", txt_path)
