"""
QQQ·SOXL 옵션체인(행사가별 미결제약정·내재변동성)을 받아
넷감마(GEX)·제로감마 플립·콜월·풋월·맥스페인을 계산하고
data/summary.json 의 "options" 항목과 data/options/<종목>.json 에 기록한다.
scripts/fetch.py 다음에 GitHub Actions 에서 실행된다.

계산 방식 (OptionCharts 등 일반적인 무료 GEX 사이트와 같은 가정)
- 딜러는 콜을 들고(+감마) 풋을 판다(−감마)고 가정한다.
- 행사가별 GEX = 감마 × OI × 100 × 현재가² × 0.01  (현재가 1% 움직일 때 딜러 헤지 달러 규모)
  콜은 +, 풋은 − 로 더한다.
- 플립 = 가격을 위아래로 옮겨 가며 전체 넷감마를 다시 계산했을 때 부호가 바뀌는 가격(현재가에 가장 가까운 것).
- 콜월 = 콜 GEX 최대 행사가, 풋월 = 풋 GEX(절대값) 최대 행사가.
- OI 는 거래소가 장 시작 전에 갱신하므로 "전 거래일 장 시작 시점" 물량이다.
사이트(firmtape, OptionCharts)마다 가정이 달라 플립이 1~2달러 차이 날 수 있다.
"""
import json
import math
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

SYMBOLS = ["QQQ", "SOXL"]
MAX_DAYS = 90          # 합산에 넣을 최대 만기(일)
NEAR_COUNT = 3         # 따로 보여줄 가까운 만기 개수
RATE = 0.04            # 무위험 금리 가정
ET = ZoneInfo("America/New_York")


def bs_gamma(S, K, T, iv):
    """블랙숄즈 감마 (배열 연산). S 는 스칼라 또는 배열."""
    S = np.asarray(S, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (RATE + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
        g = np.exp(-0.5 * d1**2) / (np.sqrt(2 * np.pi) * S * iv * np.sqrt(T))
    return np.nan_to_num(g)


def clean_iv(df: pd.DataFrame, fallback: float) -> pd.Series:
    """야후 IV 는 OI 가 없는 행사가에서 0 근처 엉터리 값이 많아 보정한다."""
    iv = df["impliedVolatility"].astype(float).where(lambda x: (x > 0.03) & (x < 5))
    iv = iv.interpolate(limit_direction="both")
    return iv.fillna(fallback)


def load_chain(tk: yf.Ticker, exp: str, spot: float, now: datetime):
    ch = tk.option_chain(exp)
    exp_dt = datetime.strptime(exp, "%Y-%m-%d").replace(hour=16, tzinfo=ET)
    T = max((exp_dt - now).total_seconds() / (365 * 24 * 3600), 1 / (365 * 24))  # 최소 1시간
    rows = []
    for side, df in (("C", ch.calls), ("P", ch.puts)):
        if df is None or df.empty:
            continue
        df = df.sort_values("strike").reset_index(drop=True)
        atm = df.iloc[(df["strike"] - spot).abs().argsort()[:4]]["impliedVolatility"]
        atm = atm[(atm > 0.03) & (atm < 5)]
        fallback = float(atm.median()) if len(atm) else 0.5
        out = pd.DataFrame(
            {
                "strike": df["strike"].astype(float),
                "oi": df["openInterest"].fillna(0).astype(float),
                "vol": df["volume"].fillna(0).astype(float),
                "iv": clean_iv(df, fallback),
            }
        )
        out["side"] = side
        rows.append(out)
    chain = pd.concat(rows, ignore_index=True)
    chain["exp"] = exp
    chain["T"] = T
    return chain


def gex_at(chain: pd.DataFrame, S) -> np.ndarray:
    """가격 S(배열)에서의 전체 넷감마($/1%)"""
    S = np.atleast_1d(np.asarray(S, dtype=float))[:, None]
    g = bs_gamma(S, chain["strike"].values[None, :], chain["T"].values[None, :], chain["iv"].values[None, :])
    sign = np.where(chain["side"].values == "C", 1.0, -1.0)[None, :]
    return (g * chain["oi"].values[None, :] * 100 * S**2 * 0.01 * sign).sum(axis=1)


def find_flip(chain: pd.DataFrame, spot: float):
    grid = np.linspace(spot * 0.7, spot * 1.3, 601)
    tot = gex_at(chain, grid)
    s = np.sign(tot)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    if not len(idx):
        return None
    # 선형보간으로 0 이 되는 가격, 현재가에 가장 가까운 것
    flips = [grid[i] - tot[i] * (grid[i + 1] - grid[i]) / (tot[i + 1] - tot[i]) for i in idx]
    return float(min(flips, key=lambda x: abs(x - spot)))


def max_pain(chain: pd.DataFrame):
    strikes = np.sort(chain["strike"].unique())
    c = chain[chain["side"] == "C"]
    p = chain[chain["side"] == "P"]
    pain = [
        (np.maximum(k - c["strike"].values, 0) * c["oi"].values).sum()
        + (np.maximum(p["strike"].values - k, 0) * p["oi"].values).sum()
        for k in strikes
    ]
    return float(strikes[int(np.argmin(pain))])


def r(x, nd=2):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


def analyze(chain: pd.DataFrame, spot: float, label: str) -> dict:
    ch = chain[chain["oi"] > 0].copy()
    sign = np.where(ch["side"] == "C", 1.0, -1.0)
    ch["gex"] = bs_gamma(spot, ch["strike"].values, ch["T"].values, ch["iv"].values) * ch["oi"] * 100 * spot**2 * 0.01 * sign
    by = ch.groupby(["strike", "side"])["gex"].sum().unstack(fill_value=0.0)
    for col in ("C", "P"):
        if col not in by:
            by[col] = 0.0
    by["net"] = by["C"] + by["P"]
    oi = ch.groupby(["strike", "side"])["oi"].sum().unstack(fill_value=0.0)
    net_total = float(by["net"].sum())
    flip = find_flip(ch, spot)
    call_wall = float(by["C"].idxmax()) if by["C"].max() > 0 else None
    put_wall = float(by["P"].idxmin()) if by["P"].min() < 0 else None

    def top(series, n, asc=False):
        s = series.sort_values(ascending=asc).head(n)
        return [[r(k), r(v, 0)] for k, v in s.items()]

    # 현재가 위아래 가까운 큰 물량(감마 기준) 상위
    above = by[by.index > spot]["net"]
    below = by[by.index < spot]["net"]
    return {
        "구분": label,
        "넷감마($/1%)": r(net_total, 0),
        "감마부호": "양수" if net_total > 0 else "음수",
        "콜GEX합": r(float(by["C"].sum()), 0),
        "풋GEX합": r(float(by["P"].sum()), 0),
        "플립": r(flip),
        "현재가_플립위": None if flip is None else bool(spot > flip),
        "콜월": r(call_wall),
        "풋월": r(put_wall),
        "맥스페인": r(max_pain(ch)) if ch["exp"].nunique() == 1 else None,
        "위쪽_양감마_상위(행사가,GEX)": top(above[above > 0], 4),
        "아래쪽_음감마_상위(행사가,GEX)": top(below[below < 0], 4, asc=True),
        "콜OI_상위(행사가,OI)": top(oi.get("C", pd.Series(dtype=float)), 5),
        "풋OI_상위(행사가,OI)": top(oi.get("P", pd.Series(dtype=float)), 5),
        "풋콜OI비율": r(float(oi.get("P", pd.Series([0])).sum() / max(oi.get("C", pd.Series([1])).sum(), 1)), 2),
    }


def run(symbol: str, now: datetime) -> dict:
    tk = yf.Ticker(symbol)
    hist = tk.history(period="5d", auto_adjust=False)
    spot = float(hist["Close"].iloc[-1])
    spot_date = hist.index[-1].strftime("%Y-%m-%d")
    today_et = now.astimezone(ET).date()
    exps = []
    for e in tk.options:
        d = datetime.strptime(e, "%Y-%m-%d").date()
        # 이미 만기된 것(장 마감 후 실행 시 당일 만기 포함) 제외
        if d < today_et or (d == today_et and now.astimezone(ET).hour >= 16):
            continue
        if (d - today_et).days <= MAX_DAYS:
            exps.append(e)
    chains = []
    for e in exps:
        try:
            chains.append(load_chain(tk, e, spot, now))
        except Exception as ex:  # 한 만기 실패로 전체를 멈추지 않는다
            print(f"{symbol} {e} 실패: {ex}")
    if not chains:
        return {"오류": "옵션체인을 받지 못함"}
    allc = pd.concat(chains, ignore_index=True)
    # 행사가가 현재가에서 너무 먼 것은 감마가 0 이라 계산에서 빼도 같다 (속도용)
    allc = allc[(allc["strike"] > spot * 0.5) & (allc["strike"] < spot * 1.5)]

    res = {
        "기준가": r(spot),
        "기준가_날짜": spot_date,
        "OI_기준": "전 거래일 장 시작 시점 미결제약정",
        "만기별": [],
    }
    for e in exps[:NEAR_COUNT]:
        c = allc[allc["exp"] == e]
        if len(c):
            dte = (datetime.strptime(e, "%Y-%m-%d").date() - today_et).days
            res["만기별"].append({"만기": e, "DTE": dte, **analyze(c, spot, f"{e} 단독")})
    week = [e for e in exps if (datetime.strptime(e, "%Y-%m-%d").date() - today_et).days <= 9]
    if week:
        res["1주합산"] = {"만기": week, **analyze(allc[allc["exp"].isin(week)], spot, "9일 이내 만기 합산")}
    res["전체합산"] = {"만기수": len(exps), **analyze(allc, spot, f"{MAX_DAYS}일 이내 전체 만기 합산")}
    return res


def main():
    now = datetime.now(timezone.utc)
    out = {"generated_at_utc": now.strftime("%Y-%m-%d %H:%M")}
    os.makedirs("data/options", exist_ok=True)
    for s in SYMBOLS:
        try:
            out[s] = run(s, now)
        except Exception as ex:
            out[s] = {"오류": str(ex)}
        with open(f"data/options/{s}.json", "w", encoding="utf-8") as f:
            json.dump(out[s], f, ensure_ascii=False, indent=2)
    path = "data/summary.json"
    summary = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    summary["options"] = out
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({s: {k: v for k, v in out[s].items() if k in ("기준가", "전체합산", "1주합산")} for s in SYMBOLS}, ensure_ascii=False, indent=1)[:6000])


if __name__ == "__main__":
    main()
