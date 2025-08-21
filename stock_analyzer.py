# -*- coding: utf-8 -*-
"""
AI 市場情報分析助理（台股版｜含模糊搜尋）
- 市場固定台股，輸入股票「名稱或代碼」即可模糊搜尋（Yahoo Finance Search API）
- 只列出台股 .TW / .TWO，選擇後自動填入正確代碼
- 標題顯示：公司中文名（代號）
- 其餘功能：K線、SMA20/60、支撐/壓力、量能、KD/MACD、新聞情緒與報告、下載不閃畫面
"""

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
VOLUME_LOOKBACK = 30              # 量能基準視窗
VOL_SPIKE_MULTIPLIER = 1.5        # 視為放量的倍率


# ====== 版面 ======
st.set_page_config(page_title="AI 市場情報分析助理（台股）", page_icon="📊", layout="wide")
st.markdown("""
<style>
section[data-testid="stSidebar"] { width: 300px !important; }
.header-title { font-size:26px; font-weight:800; margin: 0 0 6px 0; }
.header-sub   { color:#888; margin: 0 0 16px 0; }
.card { border:1px solid #e5e7eb; border-radius:12px; padding:14px; margin:10px 0; background:#fff; box-shadow: 0 1px 2px rgba(0,0,0,.04); }
.badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; margin-right:6px; }
.badge-green{ background:#e8f5e9; color:#1b5e20;}
.badge-red  { background:#ffebee; color:#b71c1c;}
.badge-gray { background:#eceff1; color:#37474f;}
.small { color:#666; font-size:12px; }
h2 { margin-top: 1.1rem; }
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="header-title">AI Stock Analyzer | AI 市場情報分析助理（台股）</div>', unsafe_allow_html=True)
st.markdown('<div class="header-sub">資訊整合 ｜ 自動分析 ｜ 決策輔助（不提供投資建議）</div>', unsafe_allow_html=True)


# ====== ENV / Gemini ======
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    try:
        API_KEY = st.secrets["GEMINI_API_KEY"]
    except Exception:
        API_KEY = None

if not API_KEY:
    st.error("❌ 找不到 GEMINI_API_KEY，請在 .env 或 st.secrets 設定（例如：GEMINI_API_KEY=xxxx）")
    st.stop()

@st.cache_resource(show_spinner=False)
def get_gemini_client(api_key: str):
    return genai.Client(api_key=api_key)

client = get_gemini_client(API_KEY)


# ====== Session：記住最後一次分析參數，避免下載造成畫面消失 ======
if "last_params" not in st.session_state:
    st.session_state.last_params = None

if "last_display_name" not in st.session_state:
    st.session_state.last_display_name = ""


# ====== Yahoo Finance 搜尋（台股 .TW/.TWO） ======
@st.cache_data(ttl=3600, show_spinner=False)
def yahoo_search_tw(query: str, max_items: int = 20) -> list[dict]:
    """以 Yahoo Finance Search API 模糊搜尋，僅回傳台股 .TW / .TWO"""
    if not query or len(query.strip()) < 1:
        return []

    url = "https://query1.finance.yahoo.com/v1/finance/search"
    params = {
        "q": query.strip(),
        "lang": "zh-TW",
        "region": "TW",
    }
    try:
        r = requests.get(url, params=params, timeout=7)
        r.raise_for_status()
        data = r.json() or {}
    except Exception:
        return []

    out = []
    for it in (data.get("quotes") or []):
        sym = (it.get("symbol") or "").strip()
        name = it.get("shortname") or it.get("longname") or it.get("name") or ""
        if sym.endswith(".TW") or sym.endswith(".TWO"):
            out.append({"symbol": sym, "name": name})
    # 去重 & 最多 max_items
    seen = set()
    uniq = []
    for x in out:
        if x["symbol"] not in seen:
            uniq.append(x)
            seen.add(x["symbol"])
    return uniq[:max_items]


@st.cache_data(ttl=3600, show_spinner=False)
def get_name_by_symbol(symbol: str) -> str:
    """補充：若只有代號，嘗試用 Yahoo 搜尋把名稱抓出來。"""
    if not symbol:
        return ""
    # 用代號反查
    res = yahoo_search_tw(symbol, 5)
    for it in res:
        if it["symbol"].upper() == symbol.upper():
            return it["name"] or ""
    # 退而求其次：用 yfinance.info（較慢，且有機率為空）
    try:
        info = yf.Ticker(symbol).info or {}
        return info.get("longName") or info.get("shortName") or ""
    except Exception:
        return ""


# ====== Sidebar（台股專用）======
with st.sidebar:
    st.markdown("**基本設定**")
    st.caption("市場固定：台股（.TW / .TWO）")

    # 1) 文字輸入：股票名稱 或 代碼（模糊）
    query = st.text_input("輸入名稱或代碼（支援模糊）", value="2330")

    # 2) 動態搜尋（即時）
    found = yahoo_search_tw(query, max_items=20) if query else []

    # 如果直接輸入 4 碼數字，預設補 .TW 作為候選
    if re.fullmatch(r"\d{4}", str(query).strip()):
        default_sym = f"{query.strip()}.TW"
        # 若搜尋結果裡沒有，插到最前面
        if not any(it["symbol"].upper() == default_sym.upper() for it in found):
            found.insert(0, {"symbol": default_sym, "name": ""})

    options = [f'{it["symbol"]} — {it["name"]}' if it["name"] else it["symbol"] for it in found] or ["（無結果，請輸入其他關鍵字）"]
    sel = st.selectbox("搜尋結果", options, index=0)

    # 解析使用者選到的 symbol
    if found:
        sel_idx = options.index(sel)
        symbol = found[sel_idx]["symbol"]
        display_name = found[sel_idx]["name"] or ""
    else:
        # 沒有找到，仍嘗試把輸入轉成 symbol
        symbol = f"{query.strip()}.TW" if re.fullmatch(r"\d{4}", str(query).strip()) else "2330.TW"
        display_name = ""

    # 若名稱空的話，補抓名稱
    if not display_name:
        display_name = get_name_by_symbol(symbol)

    period = st.selectbox("資料期間", ["3mo", "6mo", "1y", "2y"], index=2)
    interval = st.selectbox("K 線週期", ["1d", "1wk"], index=0)
    lookback = st.slider("技術面觀察視窗（近 N 根）", 20, 120, 30, 5)

    st.markdown("---")
    st.markdown("**AI 設定**")
    st.caption("模型固定使用 gemini-1.5-flash（更省額度、速度較快）。")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.4, 0.1)

    st.markdown("---")
    auto_news = st.toggle("自動抓取新聞（Yahoo Finance）", value=True)
    max_news = st.slider("最多分析新聞篇數", 3, 30, 10, 1)

    st.markdown("---")
    disclaimer = st.checkbox("我了解本工具僅供教育用途，非投資建議。", value=True)
    run_btn = st.button("產生分析", use_container_width=True)


# ====== 工具函式 ======
def _s(obj, default=""):
    return str(obj) if obj is not None else default

def _epoch_to_str(sec: int):
    try:
        return dt.datetime.utcfromtimestamp(int(sec)).strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return ""

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
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = pd.to_numeric(s.squeeze(), errors="coerce")
    return s

@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SEC)
def fetch_ohlcv(sym: str, per: str, itv: str) -> pd.DataFrame:
    return yf.download(sym, period=per, interval=itv, auto_adjust=False, progress=False)

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
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = macd.macd_diff()

    df["RSI14"] = RSIIndicator(close=close_s, window=14, fillna=False).rsi()

    bb = BollingerBands(close=close_s, window=20, window_dev=2, fillna=False)
    df["BB_low"]  = bb.bollinger_lband()
    df["BB_mid"]  = bb.bollinger_mavg()
    df["BB_high"] = bb.bollinger_hband()

    stoch = StochasticOscillator(high=high_s, low=low_s, close=close_s, window=9, smooth_window=3, fillna=False)
    df["KD_K"] = stoch.stoch()
    df["KD_D"] = stoch.stoch_signal()
    return df

def summarize_features(df: pd.DataFrame, lookback: int) -> dict:
    tail = df.dropna().tail(max(lookback, VOLUME_LOOKBACK)).copy()
    last, first = tail.iloc[-1], tail.iloc[0]
    pct_change = round((last["Close"]/first["Close"]-1)*100, 2)
    ma_trend = "多頭" if last["SMA20"] > last["SMA60"] else "空頭"
    golden_cross = bool((tail["SMA20"].iloc[-2] <= tail["SMA60"].iloc[-2]) and (last["SMA20"] > last["SMA60"]))
    dead_cross   = bool((tail["SMA20"].iloc[-2] >= tail["SMA60"].iloc[-2]) and (last["SMA20"] < last["SMA60"]))
    rsi_state = "超買(>70)" if last["RSI14"] >= 70 else ("超賣(<30)" if last["RSI14"] <= 30 else "中性")
    macd_state = "多頭(柱體>0)" if last.get("MACD_hist", 0) > 0 else "空頭(柱體<0)"

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
        "ma_trend": ma_trend,
        "golden_cross": golden_cross,
        "dead_cross": dead_cross,
        "rsi14": round(float(last["RSI14"]), 2) if pd.notna(last["RSI14"]) else None,
        "rsi_state": rsi_state,
        "macd_hist": round(float(last.get("MACD_hist", float('nan'))), 4) if pd.notna(last.get("MACD_hist", None)) else None,
        "macd_state": macd_state,
        "bb_mid": round(float(last.get("BB_mid", float('nan'))), 2) if pd.notna(last.get("BB_mid", None)) else None,
        "volume_last": int(last["Volume"]),
        "volume_mean_period": int(vol_mean),
        "volume_spike": vol_spike,
        "support_near": support,
        "resistance_near": resistance,
        "kd_k": round(float(last.get("KD_K", float('nan'))), 2) if pd.notna(last.get("KD_K", None)) else None,
        "kd_d": round(float(last.get("KD_D", float('nan'))), 2) if pd.notna(last.get("KD_D", None)) else None,
        "macd": round(float(last.get("MACD", float('nan'))), 4) if pd.notna(last.get("MACD", None)) else None,
        "macd_signal": round(float(last.get("MACD_signal", float('nan'))), 4) if pd.notna(last.get("MACD_signal", None)) else None
    }

def make_candlestick_with_volume(df: pd.DataFrame, support: float, resistance: float) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="K"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA60"], name="SMA60", mode="lines"), row=1, col=1)

    s20, s60 = df["SMA20"], df["SMA60"]
    cross_up_idx = (s20.shift(1) <= s60.shift(1)) & (s20 > s60)
    cross_dn_idx = (s20.shift(1) >= s60.shift(1)) & (s20 < s60)
    fig.add_trace(go.Scatter(x=df.index[cross_up_idx], y=df.loc[cross_up_idx, "Close"], mode="markers",
                             name="黃金交叉", marker_symbol="triangle-up", marker_size=10), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index[cross_dn_idx], y=df.loc[cross_dn_idx, "Close"], mode="markers",
                             name="死亡交叉", marker_symbol="triangle-down", marker_size=10), row=1, col=1)

    def _hline(yval, color):
        if yval is None or pd.isna(yval): return
        fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=yval, y1=yval,
                      xref="x", yref="y", line=dict(dash="dot", width=1.2, color=color))
    _hline(support, "#10b981")
    _hline(resistance, "#ef4444")

    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"), row=2, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False, height=540,
                      margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def make_kd_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["KD_K"], mode="lines", name="%K(9)"))
    fig.add_trace(go.Scatter(x=df.index, y=df["KD_D"], mode="lines", name="%D(3)"))
    fig.add_hline(y=80, line_dash="dot")
    fig.add_hline(y=20, line_dash="dot")
    if len(df.index) > 0:
        fig.add_trace(go.Scatter(x=[df.index[-1]], y=[df["KD_K"].iloc[-1]], mode="markers", name="K_last"))
        fig.add_trace(go.Scatter(x=[df.index[-1]], y=[df["KD_D"].iloc[-1]], mode="markers", name="D_last"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
    return fig

def make_macd_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Hist"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines", name="MACD"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], mode="lines", name="Signal"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
    return fig

def load_prompt(path: str, fallback: str) -> str:
    p = Path(path)
    if p.exists():
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            pass
    return fallback

def extract_json_block(text: str):
    if not text:
        return None
    m = re.search(r'```json\s*(\[[\s\S]*?\])\s*```', text) or re.search(r'(\[[\s\S]*\])', text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m2 = re.search(r'```json\s*({[\s\S]*?})\s*```', text) or re.search(r'({[\s\S]*})', text)
    if m2:
        try:
            return json.loads(m2.group(1))
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=600)
def fetch_recent_news(symbol: str, max_items: int):
    try:
        tkr = yf.Ticker(symbol)
        raw = tkr.news or []
    except Exception:
        raw = []
    items = []
    for n in raw[:max_items]:
        title = to_text(n.get("title") or "")
        link  = to_text(n.get("link") or n.get("url") or "")
        ts    = n.get("providerPublishTime") or n.get("published_on")
        pub   = _epoch_to_str(ts) if ts else ""
        desc_raw = n.get("summary") or n.get("content") or n.get("publisher") or ""
        desc  = to_text(desc_raw).strip()
        items.append({
            "title": title,
            "time": pub,
            "link": link,
            "content": desc[:NEWS_ITEM_MAX_CHARS]
        })
    return items

def classify_news_with_gemini(symbol: str, news_items: list, temperature: float):
    if not news_items:
        return []
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
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    out = []
    for it in news_items:
        out.append({
            "title": it.get("title",""),
            "published_at": it.get("time",""),
            "link": it.get("link",""),
            "summary": (it.get("content","") or "")[:200],
            "stock_sentiment": {"label":"Neutral","score":0.5},
            "article_sentiment": {"label":"Neutral","score":0.5},
            "relevance": 0.5
        })
    return out

def aggregate_sentiment(results: list):
    if not results:
        return {"total":0,"bullish":0,"neutral":0,"bearish":0,
                "bullish_ratio":0.0,"avg_article_score":0.0,"avg_relevance":0.0}
    label_map = {"Bullish":"bullish","Bearish":"bearish","Neutral":"neutral"}
    counts = {"bullish":0,"neutral":0,"bearish":0}
    s_score = s_rel = 0.0
    for r in results:
        lab = r.get("stock_sentiment",{}).get("label","Neutral")
        counts[label_map.get(lab,"neutral")] += 1
        s_score += float(r.get("article_sentiment",{}).get("score",0.5) or 0.0)
        s_rel   += float(r.get("relevance",0.0) or 0.0)
    total = len(results)
    return {
        "total": total,
        "bullish": counts["bullish"],
        "neutral": counts["neutral"],
        "bearish": counts["bearish"],
        "bullish_ratio": round(counts["bullish"]/total, 3),
        "avg_article_score": round(s_score/total, 3),
        "avg_relevance": round(s_rel/total, 3)
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
        "symbol": symbol,
        "features": features,
        "sentiment_stats": agg,
        "top_news": [
            {
                "title": r.get("title",""),
                "sentiment": r.get("stock_sentiment",{}).get("label","Neutral"),
                "relevance": r.get("relevance", 0.0),
                "summary": to_text(r.get("summary",""))
            } for r in (results or [])[:5]
        ]
    }
    prompt = f"{tpl}\n\n[輸入資料]\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    try:
        cfg = types.GenerateContentConfig(temperature=temperature)
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=[prompt], config=cfg)
        return resp.text or ""
    except Exception as e:
        return f"（AI 報告產生失敗：{e}）"

def make_report_download(name: str, text: str):
    bio = io.BytesIO(text.encode("utf-8"))
    st.download_button("⬇️ 下載 Markdown 報告", data=bio, file_name=name, mime="text/markdown")


# ====== 主流程（包成函式，下載 rerun 時用相同參數重畫） ======
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

    if not disclaimer:
        st.warning("請勾選『僅供教育用途』後再執行。")
        st.stop()

    # 大標題：公司名（代號）
    if display_name:
        st.markdown(f"### {display_name}（ {symbol} ）")
    else:
        st.markdown(f"### （ {symbol} ）")

    # 1) 價格與指標
    data = fetch_ohlcv(symbol, period, interval)
    if data.empty:
        st.warning("抓不到資料，請確認代碼或期間設定。")
        st.stop()
    df = compute_indicators(data)
    feats = summarize_features(df, lookback=lookback)

    st.markdown("## 一、股價趨勢圖")
    fig = make_candlestick_with_volume(df, feats["support_near"], feats["resistance_near"])
    st.plotly_chart(fig, use_container_width=True)

    # KPI 第一排
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最新收盤", feats["close_last"])
    c2.metric("近N根漲跌幅(%)", feats["pct_change_period"], help="以畫面上方『技術面觀察視窗』設定之根數計算")
    c3.metric("支撐區", feats["support_near"])
    c4.metric("壓力區", feats["resistance_near"])

    # KPI 第二排
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("RSI14", _s(feats["rsi14"]), help="相對強弱指標，>70 偏熱、<30 偏冷。")
    c6.metric("量能是否放大", "是" if feats["volume_spike"] else "否",
              help=f"最新一根成交量 vs 近 {VOLUME_LOOKBACK} 根均量；若 ≥ {VOL_SPIKE_MULTIPLIER} 倍視為放大。")
    c7.metric("量能基準(近30日均量)", f"{feats['volume_mean_period']:,}",
              help=f"近 {VOLUME_LOOKBACK} 根成交量平均值。")
    c8.metric("今日量能", f"{feats['volume_last']:,}",
              help="最新一根 K 的成交量；若週期為 1d 即為今日量。")

    # 技術輔助指標
    st.markdown("### 技術輔助指標")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("KD %K(9,3)", _s(feats["kd_k"]))
    k2.metric("KD %D(9,3)", _s(feats["kd_d"]))
    if feats["macd"] is not None and feats["macd_signal"] is not None:
        k3.metric("MACD / Signal", f"{feats['macd']} / {feats['macd_signal']}")
    else:
        k3.metric("MACD / Signal", "—")
    k4.metric("MACD Hist", _s(feats["macd_hist"]))

    st.markdown("### KD 與 MACD 圖形觀察")
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(make_kd_chart(df.dropna()), use_container_width=True)
    with g2:
        st.plotly_chart(make_macd_chart(df.dropna()), use_container_width=True)

    # 2) 新聞 + 情緒
    results = []
    agg = aggregate_sentiment([])
    st.markdown("## 二、情報新聞摘要")
    if auto_news:
        with st.spinner("抓取新聞中…"):
            items = fetch_recent_news(symbol, max_news)
        if items:
            with st.spinner("AI 情緒分析中…"):
                results = classify_news_with_gemini(symbol, items, temperature)
            agg = aggregate_sentiment(results)

            r1, r2, r3, r4 = st.columns([2,1,1,1])
            with r2: st.metric("新聞總篇數", agg["total"])
            with r3: st.metric("正向比例", f'{int(agg["bullish_ratio"]*100)}%')
            with r4: st.metric("平均相關性分數", agg["avg_relevance"])

            pie_df = pd.DataFrame({
                "label":["正向(Bullish)", "中立(Neutral)", "負向(Bearish)"],
                "value":[agg["bullish"], agg["neutral"], agg["bearish"]]
            })
            st.plotly_chart(
                px.pie(pie_df, names="label", values="value", hole=0.55, title="情緒統計結果")
                  .update_layout(margin=dict(l=0,r=0,t=40,b=0), height=340),
                use_container_width=True
            )

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
            st.info("目前抓不到新聞（Yahoo 來源可能暫無資料或 API 限制），已跳過情緒分析。")
    else:
        st.info("已關閉自動抓新聞。")

    # 3) 綜合報告 + 下載
    st.markdown("## 三、市場情緒總結  ＆  四、AI 分析結果")
    with st.spinner("AI 綜合分析中…"):
        report_text = build_overall_report(symbol, feats, results, agg, temperature)
    st.markdown(report_text or "（沒有產生內容）")
    make_report_download(f"{symbol}_analysis.md", report_text or "")

    st.caption("※ 本工具為教育用途的分析輔助，僅依你提供的資料與指標摘要生成，不構成投資建議。")


# ====== 事件處理：按下產生｜或下載後 rerun ======
if run_btn:
    st.session_state.last_params = {
        "symbol": symbol,
        "display_name": display_name,
        "period": period,
        "interval": interval,
        "lookback": lookback,
        "temperature": temperature,
        "auto_news": auto_news,
        "max_news": max_news,
        "disclaimer": disclaimer,
    }
    run_analysis(st.session_state.last_params)

elif st.session_state.last_params:
    # 使用上一次參數自動重畫（避免按下載後畫面消失）
    run_analysis(st.session_state.last_params)
