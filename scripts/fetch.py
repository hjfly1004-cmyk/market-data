"""
매일 장 마감 후 시세를 받아 data/*.csv 로 저장하고,
토니크로스 기준 지표를 계산해 data/summary.json 에 기록한다.
GitHub Actions 에서 실행된다.
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = {
    "VIX": "^VIX",
    "VIX3M": "^VIX3M",
    "VXX": "VXX",
    "SPX": "^GSPC",
    "SPY": "SPY",
    "QQQ": "QQQ",
    "SOXX": "SOXX",
    "SOXL": "SOXL",
}
WATCH = ["SPX", "QQQ", "SOXX", "SOXL", "SPY"]  # 진입 신호를 계산할 종목


# ───────────── 공통 지표 ─────────────
def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    w = df.resample("W-FRI").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    return w.dropna(subset=["Close"])


def completed_weeks(w: pd.DataFrame, last_date: pd.Timestamp) -> pd.DataFrame:
    # 마지막 주봉의 금요일이 아직 안 지났으면(진행 중) 제외
    if len(w) and w.index[-1] > last_date and last_date.weekday() != 4:
        return w.iloc[:-1]
    return w


def stoch(df, n=5, sk=3, sd=3):
    ll = df["Low"].rolling(n).min()
    hh = df["High"].rolling(n).max()
    raw = (df["Close"] - ll) / (hh - ll) * 100
    k = raw.rolling(sk).mean()
    d = k.rolling(sd).mean()
    return k, d


def macd(close, f=19, s=39, sig=9):
    m = close.ewm(span=f, adjust=False).mean() - close.ewm(span=s, adjust=False).mean()
    sg = m.ewm(span=sig, adjust=False).mean()
    return m, sg, m - sg


def rma(x, n):
    return x.ewm(alpha=1 / n, adjust=False).mean()


def dmi(df, n=14):
    up = df["High"].diff()
    dn = -df["Low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat(
        [df["High"] - df["Low"], (df["High"] - df["Close"].shift()).abs(), (df["Low"] - df["Close"].shift()).abs()],
        axis=1,
    ).max(axis=1)
    atr = rma(tr, n)
    pdi = 100 * rma(pd.Series(plus_dm, index=df.index), n) / atr
    mdi = 100 * rma(pd.Series(minus_dm, index=df.index), n) / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return pdi, mdi, rma(dx, n)


def mfi(df, n=14):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    mf = tp * df["Volume"]
    pos = mf.where(tp > tp.shift(), 0.0).rolling(n).sum()
    neg = mf.where(tp < tp.shift(), 0.0).rolling(n).sum()
    return 100 - 100 / (1 + pos / neg)


def last_cross(a, b):
    """a 가 b 를 마지막으로 교차한 날짜와 방향"""
    above = a > b
    flips = above.ne(above.shift()) & above.shift().notna()
    if not flips.any():
        return None
    dt = flips[flips].index[-1]
    return {"date": dt.strftime("%Y-%m-%d"), "type": "골든크로스" if above.loc[dt] else "데드크로스"}


def r(x, nd=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


# ───────────── 데이터 받기 ─────────────
data = {}
for name, tk in TICKERS.items():
    df = yf.download(tk, period="5y", interval="1d", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.to_csv(f"data/{name}.csv", float_format="%.4f")
    data[name] = df

summary = {"generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")}

# ───────────── ① 시장 국면 (VIX) ─────────────
vix = data["VIX"]
last = vix.index[-1]
vw_all = to_weekly(vix)
vw = completed_weeks(vw_all, last)
k, d = stoch(vw)
k_all, d_all = stoch(vw_all)
dead = (k < d) & (k.shift() >= d.shift()) & (k.rolling(4).max() >= 80)
dead_dates = [i.strftime("%Y-%m-%d") for i in dead[dead].index[-5:]]
last_dead = dead[dead].index[-1] if dead.any() else None
weeks_since_dead = None if last_dead is None else int(round((last - last_dead).days / 7))
riskon_window = weeks_since_dead is not None and weeks_since_dead <= 4

vxx_w = completed_weeks(to_weekly(data["VXX"]), data["VXX"].index[-1])
pdi, mdi, adx = dmi(vxx_w)

vix_close = float(vix["Close"].iloc[-1])
vix3m_close = float(data["VIX3M"]["Close"].iloc[-1])
term = vix_close / vix3m_close
backward = term > 1.0
vix_rising = bool(k.iloc[-1] > d.iloc[-1])
vxx_riskoff = bool(pdi.iloc[-1] > mdi.iloc[-1])
off_count = int(vix_rising) + int(vxx_riskoff) + int(backward) + int(vix_close >= 25)

if riskon_window and not backward:
    regime = "리스크온"
elif off_count >= 2:
    regime = "리스크오프"
elif vix_close < 20 and not vix_rising and not vxx_riskoff and not backward:
    regime = "우호"
else:
    regime = "중립"

summary["regime"] = {
    "국면": regime,
    "리스크오프_조건_충족수": off_count,
    "VIX_종가": r(vix_close),
    "VIX_날짜": last.strftime("%Y-%m-%d"),
    "VIX_구간": "지하2층(≥30)" if vix_close >= 30 else "지하1층(≥25)" if vix_close >= 25 else "경계(20~25)" if vix_close >= 20 else "20 아래" if vix_close >= 17 else "안정(<17)",
    "VIX_주봉_K_마감": r(k.iloc[-1]),
    "VIX_주봉_D_마감": r(d.iloc[-1]),
    "VIX_주봉_K_진행중": r(k_all.iloc[-1]),
    "VIX_주봉_D_진행중": r(d_all.iloc[-1]),
    "VIX_주봉_정방향(K>D)": vix_rising,
    "VIX_주봉_과열(K>=80)": bool(k.iloc[-1] >= 80),
    "최근_과열후데드_주": dead_dates,
    "데드후_경과주": weeks_since_dead,
    "리스크온_창(데드후4주)": riskon_window,
    "VXX_주봉_PDI": r(pdi.iloc[-1]),
    "VXX_주봉_MDI": r(mdi.iloc[-1]),
    "VXX_주봉_ADX": r(adx.iloc[-1]),
    "VIX/VIX3M": r(term, 3),
    "백워데이션": bool(backward),
    "VIX_최근5일": [r(x) for x in vix["Close"].iloc[-5:]],
}

# ───────────── ② 종목별 차트·추세 ─────────────
summary["symbols"] = {}
for name in WATCH:
    df = data[name]
    c = df["Close"]
    m, s, h = macd(c)
    wk = completed_weeks(to_weekly(df), df.index[-1])
    wm, ws, wh = macd(wk["Close"])
    wmfi = mfi(wk)
    wsk, wsd = stoch(wk)
    ma = {n: c.rolling(n).mean().iloc[-1] for n in (20, 50, 200, 300)}
    body = df["Open"] - df["Close"]
    atr = (df["High"] - df["Low"]).rolling(14).mean()
    big_bear = (df["Close"] < df["Open"]) & (body >= atr)
    recov = big_bear.shift() & (c >= df["Close"].shift() + 0.7 * body.shift())
    bull3 = (c > df["Open"]) & (c.shift() > df["Open"].shift()) & (c.shift(2) > df["Open"].shift(2))
    vol_spike = (df["Volume"] > df["Volume"].rolling(20).mean() * 2) & (df["Low"] <= df["Low"].rolling(20).min())
    bb_mid_w = wk["Close"].rolling(20).mean().iloc[-1]
    recent = lambda ser: [i.strftime("%Y-%m-%d") for i in ser[ser].index[-3:]]
    summary["symbols"][name] = {
        "날짜": df.index[-1].strftime("%Y-%m-%d"),
        "종가": r(c.iloc[-1]),
        "전일대비%": r((c.iloc[-1] / c.iloc[-2] - 1) * 100),
        "일봉_MACD": r(m.iloc[-1]),
        "일봉_시그널": r(s.iloc[-1]),
        "일봉_히스토그램": r(h.iloc[-1]),
        "일봉_MACD>시그널": bool(m.iloc[-1] > s.iloc[-1]),
        "일봉_마지막교차": last_cross(m, s),
        "주봉_MACD>시그널(마감)": bool(wm.iloc[-1] > ws.iloc[-1]),
        "주봉_MACD_마지막교차": last_cross(wm, ws),
        "주봉_MFI14": r(wmfi.iloc[-1]),
        "주봉_MFI_최근4주최저": r(wmfi.iloc[-4:].min()),
        "주봉_스토캐스틱_K/D": [r(wsk.iloc[-1]), r(wsd.iloc[-1])],
        "MA20": r(ma[20]), "MA50": r(ma[50]), "MA200": r(ma[200]), "MA300": r(ma[300]),
        "200MA_위": bool(c.iloc[-1] > ma[200]),
        "300MA_위": bool(c.iloc[-1] > ma[300]),
        "주봉_볼린저_중앙(반등목표)": r(bb_mid_w),
        "최근_3연속양봉": recent(bull3),
        "최근_큰음봉70%회복": recent(recov.fillna(False)),
        "최근_바닥거래량": recent(vol_spike),
        "최근5일_종가": [r(x) for x in c.iloc[-5:]],
    }

with open("data/summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps(summary["regime"], ensure_ascii=False, indent=2))
