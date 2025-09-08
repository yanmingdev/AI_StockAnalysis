# -*- coding: utf-8 -*-
import os, re, io, json, datetime as dt
from pathlib import Path

import requests
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dotenv import load_dotenv
from google import genai
from google.genai import types

# ta 指標
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands

# ====== 常數 ======
NEWS_ITEM_MAX_CHARS = 1200
CACHE_TTL_SEC = 900
GEMINI_MODEL = "gemini-1.5-flash"
VOLUME_LOOKBACK = 30
VOL_SPIKE_MULTIPLIER = 1.5

# ====== 版面 ======
st.set_page_config(page_title="AI 市場情報分析助理（台股）", page_icon="📊", layout="wide")

# 置頂導覽（頁內錨點 + 其他分頁）
st.markdown("""
<style>
section[data-testid="stSidebar"] { width: 300px !important; }

/* 置頂導覽條 */
.topbar {
  position: sticky; top: 0; z-index: 999;
  backdrop-filter: blur(6px);
  background: rgba(16,16,20,.78);
  border-bottom: 1px solid rgba(255,255,255,.08);
  margin: -1rem -1rem 12px -1rem; padding: 10px 16px;
}
.topbar .wrap { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
.topbar a {
  text-decoration: none; font-weight: 600; font-size: 14px;
  padding: 6px 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,.08);
}
.topbar a:hover { background: rgba(255,255,255,.06); }
.topbar .spacer { flex: 1; }
.header-title { font-size:26px; font-weight:800; margin: 6px 0 4px 0; }
.header-sub   { color:#aaa; margin: 0 0 12px 0; }
.card { border:1px solid #2b2f36; border-radius:12px; padding:14px; margin:10px 0; background:#111; box-shadow: 0 1px 2px rgba(0,0,0,.08); }
.badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; margin-right:6px; }
.badge-green{ background:#0f5132; color:#b6ffce;}
.badge-red  { background:#5c1b1b; color:#ffc4c4;}
.badge-gray { background:#2f3640; color:#dfe6e9;}
.small { color:#aaa; font-size:12px; }
</style>

<div class="topbar">
  <div class="wrap">
    <span>🧭 快速導覽：</span>
    <a href="#sec-chart">股價圖</a>
    <a href="#sec-tech">技術指標</a>
    <a href="#sec-news">情報新聞</a>
    <a href="#sec-summary">市場情緒總結</a>
    <a href="#sec-ai">AI 分析結果</a>
    <span class="spacer"></span>
    <a href="/02_trading">🧾 交易紀錄／報酬</a>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="header-title">AI Stock Analyzer | AI 市場情報分析助理（台股）</div>', unsafe_allow_html=True)
st.markdown('<div class="header-sub">資訊整合 ｜ 自動分析 ｜ 決策輔助（不提供投資建議）</div>', unsafe_allow_html=True)

# ====== ENV / Gemini ======
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
if not API_KEY:
    st.error("❌ 找不到 GEMINI_API_KEY，請在 .env 或 st.secrets 設定。")
    st.stop()

@st.cache_resource(show_spinner=False)
def get_gemini_client(api_key: str):
    return genai.Client(api_key=api_key)
client = get_gemini_client(API_KEY)

# ====== Session ======
if "last_params" not in st.session_state:
    st.session_state.last_params = None

# ====== 工具 ======
def _s(obj, default=""): return str(obj) if obj is not None else default
def _epoch_to_str(sec: int):
    try: return dt.datetime.utcfromtimestamp(int(sec)).strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception: return ""

def to_text(x) -> str:
    if x is None: return ""
    if isinstance(x, str): return x
    if isinstance(x, (int, float, np.number)): return str(x)
    if isinstance(x, (list, tuple, set)): return "；".join(to_text(i) for i in list(x)[:8])
    if isinstance(x, dict):
        if "content" in x and isinstance(x["content"], str): return x["content"]
        if "summary" in x and isinstance(x["summary"], str): return x["summary"]
        try: return "；".join(f"{k}:{to_text(v)}" for k, v in list(x.items())[:6])
        except Exception: return json.dumps(x, ensure_ascii=False)
    return str(x)

def get_series(df: pd.DataFrame, col: str) -> pd.Series:
    if isinstance(df.columns, pd.MultiIndex):
        if col in df.columns.get_level_values(0):
            s = df[col]
        else:
            match = [c for c in df.columns if isinstance(c, tuple) and c[0] == col]
            s = df[match[0]] if match else df.iloc[:, 0]
    else:
        s = df[col] if col in df.columns else df.iloc[:, 0]
    if isinstance(s, pd.DataFrame): s = s.iloc[:, 0]
    return pd.to_numeric(s.squeeze(), errors="coerce")

@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SEC)
def fetch_ohlcv(sym: str, per: str, itv: str) -> pd.DataFrame:
    return yf.download(sym, period=per, interval=itv, auto_adjust=False, progress=False)

@st.cache_data(ttl=3600, show_spinner=False)
def yahoo_search_tw(query: str, max_items: int = 20) -> list[dict]:
    if not query or len(query.strip()) < 1: return []
    url = "https://query1.finance.yahoo.com/v1/finance/search"
    params = {"q": query.strip(), "lang": "zh-TW", "region": "TW"}
    try:
        data = requests.get(url, params=params, timeout=7).json() or {}
    except Exception:
        return []
    out = []
    for it in (data.get("quotes") or []):
        sym = (it.get("symbol") or "").strip()
        name = it.get("shortname") or it.get("longname") or it.get("name") or ""
        if sym.endswith(".TW") or sym.endswith(".TWO"):
            out.append({"symbol": sym, "name": name})
    uniq, seen = [], set()
    for x in out:
        if x["symbol"] not in seen:
            uniq.append(x); seen.add(x["symbol"])
    return uniq[:max_items]

@st.cache_data(ttl=3600, show_spinner=False)
def get_name_by_symbol(symbol: str) -> str:
    if not symbol: return ""
    res = yahoo_search_tw(symbol, 5)
    for it in res:
        if it["symbol"].upper() == symbol.upper():
            return it["name"] or ""
    try:
        info = yf.Ticker(symbol).info or {}
        return info.get("longName") or info.get("shortName") or ""
    except Exception:
        return ""

def compute_indicators(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    open_s  = get_series(df, "Open")
    high_s  = get_series(df, "High")
    low_s   = get_series(df, "Low")
    close_s = get_series(df, "Close")
    vol_s   = get_series(df, "Volume").fillna(0)
    df = pd.DataFrame({
        "Open": open_s.values, "High": high_s.values, "Low": low_s.values,
        "Close": close_s.values, "Volume": vol_s.values
    }, index=raw.index)
    df["SMA20"] = SMAIndicator(close=close_s, window=20, fillna=False).sma_indicator()
    df["SMA60"] = SMAIndicator(close=close_s, window=60, fillna=False).sma_indicator()
    macd = MACD(close=close_s, window_slow=26, window_fast=12, window_sign=9, fillna=False)
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd.macd(), macd.macd_signal(), macd.macd_diff()
    df["RSI14"] = RSIIndicator(close=close_s, window=14, fillna=False).rsi()
    bb = BollingerBands(close=close_s, window=20, window_dev=2, fillna=False)
    df["BB_low"], df["BB_mid"], df["BB_high"] = bb.bollinger_lband(), bb.bollinger_mavg(), bb.bollinger_hband()
    stoch = StochasticOscillator(high=high_s, low=low_s, close=close_s, window=9, smooth_window=3, fillna=False)
    df["KD_K"], df["KD_D"] = stoch.stoch(), stoch.stoch_signal()
    return df

# === 預測工具 ===
def compute_atr(df: pd.DataFrame, window: int = 14) -> float:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    last_atr = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else float(close.iloc[-1] * 0.01)
    return max(last_atr, 0.0001)

def next_period_index(last_index: pd.Timestamp, steps: int, interval: str) -> list[pd.Timestamp]:
    idx, t = [], pd.Timestamp(last_index)
    if interval == "1wk":
        for i in range(1, steps + 1): idx.append(t + pd.Timedelta(weeks=i))
    else:
        d = t
        while len(idx) < steps:
            d = d + pd.Timedelta(days=1)
            if d.weekday() < 5: idx.append(d)
    return idx

def forecast_ohlc(df: pd.DataFrame, steps: int, interval: str = "1d"):
    close = df["Close"].dropna()
    ret = np.log(close).diff().dropna()
    use_n = min(max(60, 30), len(ret)) or len(ret)
    mu, sigma = float(ret.tail(use_n).mean()), float(ret.tail(use_n).std(ddof=0) or 0.0)
    last_close = float(close.iloc[-1])
    atr = compute_atr(df, window=14)
    f_idx = next_period_index(df.index[-1], steps, interval)

    opens, highs, lows, closes_med, p10_list, p90_list = [], [], [], [], [], []
    prev_close = last_close
    for s, ts in enumerate(f_idx, start=1):
        med = last_close * np.exp(mu * s)
        z = 1.2816
        p10 = last_close * np.exp(mu * s - z * sigma * np.sqrt(s))
        p90 = last_close * np.exp(mu * s + z * sigma * np.sqrt(s))
        o = prev_close
        delta = med - o
        hi = max(o, med) + 0.6 * atr + max(delta, 0) * 0.25
        lo = min(o, med) - 0.6 * atr + min(delta, 0) * 0.25

        opens.append(o); highs.append(hi); lows.append(lo); closes_med.append(med)
        p10_list.append(p10); p90_list.append(p90)
        prev_close = med

    fdf = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes_med},
                       index=pd.DatetimeIndex(f_idx, name=df.index.name))
    med = pd.Series(closes_med, index=fdf.index); p10 = pd.Series(p10_list, index=fdf.index); p90 = pd.Series(p90_list, index=fdf.index)
    return fdf, med, p10, p90

def add_forecast_traces(fig: go.Figure, fdf: pd.DataFrame, med: pd.Series, p10: pd.Series, p90: pd.Series, hist_last_index):
    if fdf.empty: return
    purple = "#A855F7"
    # 預測區直條背景
    fig.add_vrect(x0=hist_last_index, x1=fdf.index[-1], fillcolor="rgba(168,85,247,0.08)", line_width=0, row=1, col=1)
    # 預測 K
    fig.add_trace(go.Candlestick(
        x=fdf.index, open=fdf["Open"], high=fdf["High"], low=fdf["Low"], close=fdf["Close"], name="預測K",
        increasing=dict(line=dict(color=purple, width=1.2), fillcolor="rgba(168,85,247,0.45)"),
        decreasing=dict(line=dict(color=purple, width=1.2), fillcolor="rgba(168,85,247,0.45)"),
        opacity=0.85, whiskerwidth=0.3
    ), row=1, col=1)
    # 10-90% 區間 + 中位線
    fig.add_trace(go.Scatter(x=med.index, y=p90, name="P90(收盤)", mode="lines",
                             line=dict(width=1.2, dash="dashdot", color=purple)), row=1, col=1)
    fig.add_trace(go.Scatter(x=med.index, y=p10, name="P10(收盤)", mode="lines",
                             line=dict(width=1.2, dash="dot", color=purple),
                             fill="tonexty", fillcolor="rgba(168,85,247,0.12)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=med.index, y=med, name="預測收盤(中位)", mode="lines+markers",
                             line=dict(width=2.2, dash="dash", color=purple),
                             marker=dict(size=5, symbol="diamond")), row=1, col=1)

def summarize_features(df: pd.DataFrame, lookback: int) -> dict:
    tail = df.dropna().tail(max(lookback, VOLUME_LOOKBACK)).copy()
    last, first = tail.iloc[-1], tail.iloc[0]
    pct_change = round((last["Close"]/first["Close"]-1)*100, 2)
    ma_trend = "多頭" if last["SMA20"] > last["SMA60"] else "空頭"
    golden_cross = bool((tail["SMA20"].iloc[-2] <= tail["SMA60"].iloc[-2]) and (last["SMA20"] > last["SMA60"]))
    dead_cross   = bool((tail["SMA20"].iloc[-2] >= tail["SMA60"].iloc[-2]) and (last["SMA20"] < last["SMA60"]))
    rsi_state = "超買(>70)" if last["RSI14"] >= 70 else ("超賣(<30)" if last["RSI14"] <= 30 else "中性")
    vol_tail = tail["Volume"].tail(VOLUME_LOOKBACK)
    vol_mean = float(vol_tail.mean())
    vol_spike = bool(last["Volume"] >= VOL_SPIKE_MULTIPLIER * vol_mean)
    support = round(float(tail["Low"].min()), 2)
    resistance = round(float(tail["High"].max()), 2)
    return {
        "as_of": str(tail.index[-1].date()),
        "period_bars": lookback,
        "close_last": round(float(last["Close"]), 2),
        "pct_change_period": pct_change,
        "ma_trend": ma_trend, "golden_cross": golden_cross, "dead_cross": dead_cross,
        "rsi14": round(float(last["RSI14"]), 2) if pd.notna(last["RSI14"]) else None,
        "rsi_state": rsi_state,
        "volume_last": int(last["Volume"]), "volume_mean_period": int(vol_mean), "volume_spike": vol_spike,
        "support_near": support, "resistance_near": resistance,
        "kd_k": round(float(last.get("KD_K", float('nan'))), 2) if pd.notna(last.get("KD_K", None)) else None,
        "kd_d": round(float(last.get("KD_D", float('nan'))), 2) if pd.notna(last.get("KD_D", None)) else None,
        "macd": round(float(last.get("MACD", float('nan'))), 4) if pd.notna(last.get("MACD", None)) else None,
        "macd_signal": round(float(last.get("MACD_signal", float('nan'))), 4) if pd.notna(last.get("MACD_signal", None)) else None,
        "macd_hist": round(float(last.get("MACD_hist", float('nan'))), 4) if pd.notna(last.get("MACD_hist", None)) else None
    }

def make_candlestick_with_volume(df: pd.DataFrame, support: float, resistance: float) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    # 主 K
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                                 name="K",
                                 increasing=dict(line=dict(color="#26A69A", width=1.2), fillcolor="rgba(38,166,154,0.6)"),
                                 decreasing=dict(line=dict(color="#EF5350", width=1.2), fillcolor="rgba(239,83,80,0.6)")),
                  row=1, col=1)
    # SMA
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20", mode="lines",
                             line=dict(width=2.2, color="#60A5FA")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA60"], name="SMA60", mode="lines",
                             line=dict(width=2.2, color="#F9A8D4")), row=1, col=1)
    # 交叉標記
    s20, s60 = df["SMA20"], df["SMA60"]
    cross_up_idx = (s20.shift(1) <= s60.shift(1)) & (s20 > s60)
    cross_dn_idx = (s20.shift(1) >= s60.shift(1)) & (s20 < s60)
    fig.add_trace(go.Scatter(x=df.index[cross_up_idx], y=df.loc[cross_up_idx, "Close"], mode="markers",
                             name="黃金交叉", marker_symbol="triangle-up", marker_size=10, marker_color="#22c55e"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index[cross_dn_idx], y=df.loc[cross_dn_idx, "Close"], mode="markers",
                             name="死亡交叉", marker_symbol="triangle-down", marker_size=10, marker_color="#f97316"), row=1, col=1)
    # 支撐/壓力
    def _hline(yval, color):
        if yval is None or pd.isna(yval): return
        fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=yval, y1=yval,
                      xref="x", yref="y", line=dict(dash="dot", width=1.4, color=color))
    _hline(support, "#10b981"); _hline(resistance, "#ef4444")
    # 量
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="rgba(100,116,139,0.7)"), row=2, col=1)
    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=620,
        margin=dict(l=10, r=60, t=36, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    return fig

def make_kd_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["KD_K"], mode="lines", name="%K(9)"))
    fig.add_trace(go.Scatter(x=df.index, y=df["KD_D"], mode="lines", name="%D(3)"))
    fig.add_hline(y=80, line_dash="dot"); fig.add_hline(y=20, line_dash="dot")
    if len(df.index) > 0:
        fig.add_trace(go.Scatter(x=[df.index[-1]], y=[df["KD_K"].iloc[-1]], mode="markers", name="K_last"))
        fig.add_trace(go.Scatter(x=[df.index[-1]], y=[df["KD_D"].iloc[-1]], mode="markers", name="D_last"))
    fig.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
    return fig

def make_macd_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Hist"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines", name="MACD"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], mode="lines", name="Signal"))
    fig.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=10), legend=dict(orientation="h"))
    return fig

def load_prompt(path: str, fallback: str) -> str:
    p = Path(path)
    if p.exists():
        try: return p.read_text(encoding="utf-8")
        except Exception: pass
    return fallback

def extract_json_block(text: str):
    if not text: return None
    m = re.search(r'```json\s*(\[[\s\S]*?\])\s*```', text) or re.search(r'(\[[\s\S]*\])', text)
    if m:
        try: return json.loads(m.group(1))
        except Exception: pass
    m2 = re.search(r'```json\s*({[\s\S]*?})\s*```', text) or re.search(r'({[\s\S]*})', text)
    if m2:
        try: return json.loads(m2.group(1))
        except Exception: pass
    try: return json.loads(text)
    except Exception: return None

@st.cache_data(show_spinner=False, ttl=600)
def fetch_recent_news(symbol: str, max_items: int):
    try: raw = yf.Ticker(symbol).news or []
    except Exception: raw = []
    items = []
    for n in raw[:max_items]:
        title = to_text(n.get("title") or "")
        link  = to_text(n.get("link") or n.get("url") or "")
        ts    = n.get("providerPublishTime") or n.get("published_on")
        pub   = _epoch_to_str(ts) if ts else ""
        desc  = to_text(n.get("summary") or n.get("content") or n.get("publisher") or "").strip()
        items.append({"title": title, "time": pub, "link": link, "content": desc[:NEWS_ITEM_MAX_CHARS]})
    return items

def classify_news_with_gemini(symbol: str, news_items: list, temperature: float):
    if not news_items: return []
    fallback = (
        "你是金融文本標註員。僅依提供的清單對每則輸出 JSON，不要多餘說明。\n"
        "每則輸出欄位：title, published_at, link, summary,\n"
        "stock_sentiment:{label:[Bullish,Bearish,Neutral],score:0~1}, "
        "article_sentiment:{label:[Positive,Negative,Neutral],score:0~1}, relevance:0~1。\n"
        "請只輸出 JSON。"
    )
    tpl = load_prompt("news_classify_prompt.txt", fallback)
    prompt = f"{tpl}\n\n[輸入資料]\n股票: {symbol}\n新聞清單(JSON):\n{json.dumps(news_items, ensure_ascii=False, indent=2)}"
    try:
        cfg = types.GenerateContentConfig(temperature=temperature)
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=[prompt], config=cfg)
        parsed = extract_json_block(resp.text or "")
        if isinstance(parsed, list): return parsed
    except Exception: pass
    return [{
        "title": it.get("title",""), "published_at": it.get("time",""), "link": it.get("link",""),
        "summary": (it.get("content","") or "")[:200],
        "stock_sentiment": {"label":"Neutral","score":0.5},
        "article_sentiment": {"label":"Neutral","score":0.5},
        "relevance": 0.5
    } for it in (news_items or [])]

def aggregate_sentiment(results: list):
    if not results:
        return {"total":0,"bullish":0,"neutral":0,"bearish":0,"bullish_ratio":0.0,"avg_article_score":0.0,"avg_relevance":0.0}
    label_map = {"Bullish":"bullish","Bearish":"bearish","Neutral":"neutral"}
    counts = {"bullish":0,"neutral":0,"bearish":0}; s_score = s_rel = 0.0
    for r in results:
        lab = r.get("stock_sentiment",{}).get("label","Neutral")
        counts[label_map.get(lab,"neutral")] += 1
        s_score += float(r.get("article_sentiment",{}).get("score",0.5) or 0.0)
        s_rel   += float(r.get("relevance",0.0) or 0.0)
    total = len(results)
    return {
        "total": total, "bullish": counts["bullish"], "neutral": counts["neutral"], "bearish": counts["bearish"],
        "bullish_ratio": round(counts["bullish"]/total, 3),
        "avg_article_score": round(s_score/total, 3), "avg_relevance": round(s_rel/total, 3)
    }

def build_overall_report(symbol: str, features: dict, results: list, agg: dict, temperature: float):
    fallback = (
        "你是嚴謹的投資研究助理。以條列輸出，不要投資建議；"
        "必要時用『若A則B』條件化語句。\n"
        "請依據下方 JSON 輸出章節：\n"
        "## 市場情緒總結\n"
        "## AI 分析結果\n"
        "### 分析總結\n"
        "### 媒體情緒觀察\n"
        "### 條件化情境與價位帶"
    )
    tpl = load_prompt("final_report_prompt.txt", fallback)
    payload = {
        "symbol": symbol, "features": features, "sentiment_stats": agg,
        "top_news": [{
            "title": r.get("title",""),
            "sentiment": r.get("stock_sentiment",{}).get("label","Neutral"),
            "relevance": r.get("relevance", 0.0),
            "summary": to_text(r.get("summary",""))
        } for r in (results or [])[:5]]
    }
    prompt = f"{tpl}\n\n[輸入資料]\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    try:
        cfg = types.GenerateContentConfig(temperature=temperature)
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=[prompt], config=cfg)
        return resp.text or ""
    except Exception as e:
        return f"（AI 報告產生失敗：{e}）"

def make_report_download(name: str, text: str):
    st.download_button("⬇️ 下載 Markdown 報告", data=io.BytesIO(text.encode("utf-8")), file_name=name, mime="text/markdown")

# 將 AI 報告切成「市場情緒總結」與「AI 分析結果」兩段
def split_ai_report(text: str) -> tuple[str, str]:
    if not text: return "", ""
    # 找到「## AI 分析結果」的開頭
    m = re.search(r"^##\s*AI\s*分析結果.*$", text, flags=re.M)
    if not m:
        return text, ""  # 找不到就全部放第一段
    part1 = text[:m.start()]
    part2 = text[m.start():]
    # 去掉兩段各自的第一個 H2 標題，避免重複
    part1 = re.sub(r"^##\s*[^\n]+\n?", "", part1.strip(), count=1, flags=re.M)
    part2 = re.sub(r"^##\s*[^\n]+\n?", "", part2.strip(), count=1, flags=re.M)
    return part1.strip(), part2.strip()

# ====== Sidebar（台股專用）======
with st.sidebar:
    st.markdown("**基本設定**")
    st.caption("市場固定：台股（.TW / .TWO）")
    query = st.text_input("輸入名稱或代碼（支援模糊）", value="2330")

    # 搜尋候選
    found = (lambda q: [] if not q else yahoo_search_tw(q, 20))(query)
    if re.fullmatch(r"\d{4}", str(query).strip()):
        default_sym = f"{query.strip()}.TW"
        if not any(it["symbol"].upper() == default_sym.upper() for it in found):
            found.insert(0, {"symbol": default_sym, "name": ""})
    options = [f'{it["symbol"]} — {it["name"]}' if it["name"] else it["symbol"] for it in found] or ["（無結果，請輸入其他關鍵字）"]
    sel = st.selectbox("搜尋結果", options, index=0)
    if found:
        sel_idx = options.index(sel); symbol = found[sel_idx]["symbol"]; display_name = found[sel_idx]["name"] or ""
    else:
        symbol, display_name = ("2330.TW", "")
    if not display_name: display_name = get_name_by_symbol(symbol)

    period = st.selectbox("資料期間", ["3mo", "6mo", "1y", "2y"], index=2)
    interval = st.selectbox("K 線週期", ["1d", "1wk"], index=0)
    lookback = st.slider("技術面觀察視窗（近 N 根）", 20, 120, 30, 5)

    # 預測：固定一週（日K=5、週K=1）
    st.markdown("---")
    st.markdown("**實驗功能：價格路徑預測**")
    predict_enabled = st.toggle("顯示未來一週預測", value=True)
    forecast_steps = 5 if interval == "1d" else 1

    # AI 溫度
    st.markdown("---")
    st.markdown("**AI 設定**")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.4, 0.1)

    # 新聞：固定抓最新最多 10 則（只保留開關）
    st.markdown("---")
    auto_news = st.toggle("自動抓取新聞（Yahoo Finance）", value=True)
    max_news = 10

    st.markdown("---")
    disclaimer = st.checkbox("我了解本工具僅供教育用途，非投資建議。", value=True)
    run_btn = st.button("產生分析", use_container_width=True)

# ====== 主流程 ======
def run_analysis(params: dict):
    symbol       = params["symbol"]
    display_name = params.get("display_name") or get_name_by_symbol(symbol)
    period       = params["period"]
    interval     = params["interval"]
    lookback     = params["lookback"]
    temperature  = params["temperature"]
    auto_news    = params["auto_news"]
    max_news     = params["max_news"]
    disclaimer   = params["disclaimer"]
    predict_enabled = params.get("predict_enabled", False)
    forecast_steps  = int(params.get("forecast_steps", 5))

    if not disclaimer:
        st.warning("請勾選『僅供教育用途』後再執行。"); st.stop()

    # 公司抬頭
    st.markdown(f"### {display_name}（ {symbol} ）" if display_name else f"### （ {symbol} ）")

    # 資料與指標
    data = fetch_ohlcv(symbol, period, interval)
    if data.empty: st.warning("抓不到資料，請確認代碼或期間設定。"); st.stop()
    df = compute_indicators(data)
    feats = summarize_features(df, lookback=lookback)

    # --- 股價趨勢圖 ---
    st.markdown('<a id="sec-chart"></a>', unsafe_allow_html=True)
    st.markdown("## 一、股價趨勢圖")
    fig = make_candlestick_with_volume(df, feats["support_near"], feats["resistance_near"])
    if predict_enabled and forecast_steps > 0:
        try:
            base = df[["Open","High","Low","Close"]].dropna()
            fdf, med, p10, p90 = forecast_ohlc(base, steps=forecast_steps, interval=interval)
            add_forecast_traces(fig, fdf, med, p10, p90, hist_last_index=df.index[-1])
            fig.update_xaxes(range=[df.index[0], fdf.index[-1] + pd.Timedelta(days=2)])
        except Exception as e:
            st.info(f"（預測層載入失敗：{e}）")
    st.plotly_chart(fig, use_container_width=True)

    # KPI
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最新收盤", feats["close_last"])
    c2.metric("近N根漲跌幅(%)", feats["pct_change_period"])
    c3.metric("支撐區", feats["support_near"])
    c4.metric("壓力區", feats["resistance_near"])

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("RSI14", _s(feats["rsi14"]))
    c6.metric("量能是否放大", "是" if feats["volume_spike"] else "否",
              help=f"最新一根 vs 近 {VOLUME_LOOKBACK} 根均量；≥ {VOL_SPIKE_MULTIPLIER} 倍視為放大")
    c7.metric("量能基準(近30日)", f"{feats['volume_mean_period']:,}")
    c8.metric("最新量", f"{feats['volume_last']:,}")

    # --- 技術輔助指標 ---
    st.markdown('<a id="sec-tech"></a>', unsafe_allow_html=True)
    st.markdown("### 技術輔助指標")
    col_a, col_b = st.columns(2)
    with col_a: st.plotly_chart(make_kd_chart(df.dropna()), use_container_width=True)
    with col_b: st.plotly_chart(make_macd_chart(df.dropna()), use_container_width=True)

    # --- 情報新聞摘要 ---
    st.markdown('<a id="sec-news"></a>', unsafe_allow_html=True)
    st.markdown("## 二、情報新聞摘要")
    results = []; agg = aggregate_sentiment([])
    if auto_news:
        with st.spinner("抓取新聞中…"): items = fetch_recent_news(symbol, max_news)
        if items:
            with st.spinner("AI 情緒分析中…"): results = classify_news_with_gemini(symbol, items, temperature)
            agg = aggregate_sentiment(results)
            r1, r2, r3, r4 = st.columns([2,1,1,1])
            with r2: st.metric("新聞總篇數", agg["total"])
            with r3: st.metric("正向比例", f'{int(agg["bullish_ratio"]*100)}%')
            with r4: st.metric("平均相關性", agg["avg_relevance"])
            pie_df = pd.DataFrame({"label":["正向(Bullish)","中立(Neutral)","負向(Bearish)"],
                                   "value":[agg["bullish"],agg["neutral"],agg["bearish"]]})
            st.plotly_chart(px.pie(pie_df, names="label", values="value", hole=0.55, title="情緒統計結果")
                            .update_layout(template="plotly_dark", margin=dict(l=0,r=0,t=40,b=0), height=340),
                            use_container_width=True)
            st.markdown("### 🧾 詳細新聞列表")
            for r in results:
                stock_lab = r.get("stock_sentiment",{}).get("label","Neutral")
                art_lab   = r.get("article_sentiment",{}).get("label","Neutral")
                stock_badge = "badge-green" if stock_lab=="Bullish" else ("badge-red" if stock_lab=="Bearish" else "badge-gray")
                art_badge   = "badge-green" if art_lab=="Positive" else ("badge-red" if art_lab=="Negative" else "badge-gray")
                title = _s(r.get("title","(無標題)"))
                summary_txt = to_text(r.get('summary','')).strip()
                link_txt = to_text(r.get('link','')).strip()
                st.markdown(f"""
<div class="card">
  <div style="font-weight:600">{title}</div>
  <div class="small" style="margin:6px 0;">
    發布時間：{_s(r.get('published_at',''))}　|　相關性分數：{_s(r.get('relevance',0))}
  </div>
  <div style="margin:6px 0;">
    <span class="badge {stock_badge}">📈 股票情緒：{stock_lab}</span>
    <span class="badge {art_badge}">📰 文章情緒：{art_lab}</span>
  </div>
  <div style="white-space:pre-wrap; margin-top:6px;">{summary_txt}</div>
  {"<div style='margin-top:8px;'><a href=\"%s\" target=\"_blank\">查看原文</a></div>" % link_txt if link_txt else ""}
</div>
""", unsafe_allow_html=True)
        else:
            st.info("目前抓不到新聞，已跳過情緒分析。")
    else:
        st.info("已關閉自動抓新聞。")

    # --- 三 & 四：把 AI 報告分段顯示 ---
    st.markdown('<a id="sec-summary"></a>', unsafe_allow_html=True)
    st.markdown("## 三、市場情緒總結")
    with st.spinner("AI 生成『市場情緒總結』…"):
        full_report = build_overall_report(symbol, feats, results, agg, temperature)
        part_summary, part_ai = split_ai_report(full_report)
    st.markdown(part_summary or "（無內容）")

    st.markdown('<a id="sec-ai"></a>', unsafe_allow_html=True)
    st.markdown("## 四、AI 分析結果")
    st.markdown(part_ai or "（無內容）")

    make_report_download(f"{symbol}_analysis.md", full_report or "")

    st.caption("※ 本工具為教育用途的分析輔助，不構成投資建議。")

# ====== 事件處理（已移除多餘按鈕）======
if run_btn:
    st.session_state.last_params = {
        "symbol": symbol, "display_name": display_name,
        "period": period, "interval": interval, "lookback": lookback,
        "temperature": temperature, "auto_news": auto_news, "max_news": max_news,
        "disclaimer": True, "predict_enabled": predict_enabled, "forecast_steps": forecast_steps
    }
    run_analysis(st.session_state.last_params)
elif st.session_state.last_params:
    run_analysis(st.session_state.last_params)
