# -*- coding: utf-8 -*-
"""
AI Stock Analyzer | 台股 （精簡版：移除頁籤／交易分頁、新聞固定開啟、移除市場情緒總結顯示）
"""
import os, re, io, json, datetime as dt
from pathlib import Path
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

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

# 技術指標
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands

# ====== 常數（精簡 UI 版本的固定參數）======
NEWS_ITEM_MAX_CHARS = 1200
CACHE_TTL_SEC = 900
GEMINI_MODEL = "gemini-1.5-flash"
VOLUME_LOOKBACK = 30
VOL_SPIKE_MULTIPLIER = 1.5

# 固定圖資期間與技術視窗
DEFAULT_PERIOD = "1y"
DEFAULT_INTERVAL = "1d"
DEFAULT_LOOKBACK = 30  # 近 N 根

# 固定 AI 溫度（較詳細、較一致）
FIXED_TEMPERATURE = 0.2

# ====== 版面與樣式 ======
st.set_page_config(page_title="AI 市場情報分析助理（台股）", page_icon="📊", layout="wide")

st.markdown("""
<style>
/* 固定側欄寬度 */
section[data-testid="stSidebar"] { width: 300px !important; }

/* 隱藏 Streamlit 預設的多頁導覽（會顯示 stock analyzer / trading） */
div[data-testid="stSidebarNav"] { display: none !important; }

/* Topbar */
.topbar { position: sticky; top: 0; z-index: 999; backdrop-filter: blur(6px);
  background: rgba(16,16,20,.78); border-bottom: 1px solid rgba(255,255,255,.08);
  margin: -1rem -1rem 12px -1rem; padding: 10px 16px; }
.topbar .wrap { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
.topbar a { text-decoration: none; font-weight: 600; font-size: 14px; padding: 6px 10px;
  border-radius: 8px; border: 1px solid rgba(255,255,255,.08); }
.topbar a:hover { background: rgba(255,255,255,.06); }
.topbar .spacer { flex: 1; }

.header-title { font-size:26px; font-weight:800; margin: 6px 0 4px 0; }
.header-sub   { color:#aaa; margin: 0 0 12px 0; }

.card { border:1px solid #2b2f36; border-radius:12px; padding:14px; margin:10px 0;
  background:#111; box-shadow: 0 1px 2px rgba(0,0,0,.08); }
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
    <!-- 已移除「市場情緒總結」與「交易紀錄／報酬」連結 -->
    <a href="#sec-ai">AI 分析結果</a>
    <span class="spacer"></span>
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

# ====== 小工具 ======
def _s(obj, default=""): return str(obj) if obj is not None else default

def _epoch_to_local_str(sec: int) -> str:
    """epoch -> 本機時區 YYYY/MM/DD HH:MM"""
    try:
        return dt.datetime.fromtimestamp(int(sec)).strftime("%Y/%m/%d %H:%M")
    except Exception:
        return ""

def _fmt_dt_str(s: str) -> str:
    """RFC822/常見字串 -> 本機時區 YYYY/MM/DD HH:MM；失敗回空字串"""
    if not s:
        return ""
    for fmt in [
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S %z",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
    ]:
        try:
            d = dt.datetime.strptime(s.strip(), fmt)
            if d.tzinfo:
                return d.astimezone().strftime("%Y/%m/%d %H:%M")
            return d.strftime("%Y/%m/%d %H:%M")
        except Exception:
            continue
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
    if isinstance(s, pd.DataFrame): s = s.iloc[:, 0]
    return pd.to_numeric(s.squeeze(), errors="coerce")

# ====== 取價 / 搜尋（強化） ======
@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SEC)
def fetch_ohlcv(sym: str, per: str, itv: str) -> pd.DataFrame:
    return yf.download(sym, period=per, interval=itv, auto_adjust=False, progress=False)

@st.cache_data(ttl=600, show_spinner=False)
def _verify_symbol_has_data(symbol: str) -> bool:
    """用短期資料驗證代號是否存在（股票/ETF/權證皆可）。"""
    try:
        df = yf.download(symbol, period="5d", interval="1d", progress=False)
        return not df.empty
    except Exception:
        return False

@st.cache_data(ttl=3600, show_spinner=False)
def yahoo_search_tw(query: str, max_items: int = 20) -> list[dict]:
    """Yahoo API 模糊搜尋（公司名/代號），只回傳 .TW/.TWO。"""
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
    # 去重
    uniq, seen = [], set()
    for x in out:
        if x["symbol"] not in seen:
            uniq.append(x); seen.add(x["symbol"])
    return uniq[:max_items]

@st.cache_data(ttl=1800, show_spinner=False)
def _resolve_numeric_code_candidates(code: str) -> list[dict]:
    """
    純數字（4~6 碼）→ 依序嘗試 .TW / .TWO，驗證有價量就收錄，適用股票 / ETF / 權證。
    例：1815 → 1815.TW 或 1815.TWO；0050 → 0050.TW（ETF）。
    """
    code = code.strip()
    if not re.fullmatch(r"\d{4,6}", code):
        return []
    candidates = []
    for mkt in (".TW", ".TWO"):
        sym = f"{code}{mkt}"
        if _verify_symbol_has_data(sym):
            # 取名稱（失敗就空字串，不影響顯示）
            try:
                info = yf.Ticker(sym).info or {}
                nm = info.get("longName") or info.get("shortName") or ""
            except Exception:
                nm = ""
            candidates.append({"symbol": sym, "name": nm})
    return candidates

@st.cache_data(ttl=3600, show_spinner=False)
def get_name_by_symbol(symbol: str) -> str:
    """優先用 Yahoo 搜尋結果名稱；否則回退 yfinance 的 info 名稱；再不行就空字串。"""
    if not symbol: return ""
    try:
        info = yf.Ticker(symbol).info or {}
        nm = info.get("longName") or info.get("shortName") or ""
        if nm: return nm
    except Exception:
        pass
    res = yahoo_search_tw(symbol, 5)
    for it in res:
        if it["symbol"].upper() == symbol.upper():
            return it["name"] or ""
    return ""

# ====== 指標計算 ======
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

# ===== 簡易預測（保留） =====
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

# ====== Fibonacci（正／負 38.2% 與 61.8%，只輸出表格） ======
def compute_fib_posneg(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """以近 N 根高低的振幅 R，樞紐 P=最新收盤，輸出 ±38.2%、±61.8% 價位。"""
    tail = df.dropna().tail(lookback)
    if tail.empty:
        return pd.DataFrame()
    high = float(tail["High"].max())
    low  = float(tail["Low"].min())
    if not np.isfinite(high) or not np.isfinite(low) or high == low:
        return pd.DataFrame()
    rng = high - low
    P = float(tail["Close"].iloc[-1])
    levels = [
        ("-61.8%", round(P - 0.618*rng, 2)),
        ("-38.2%", round(P - 0.382*rng, 2)),
        ("+38.2%", round(P + 0.382*rng, 2)),
        ("+61.8%", round(P + 0.618*rng, 2)),
    ]
    df_levels = pd.DataFrame(levels, columns=["層級", "價位"])
    return df_levels

def summarize_features(df: pd.DataFrame, lookback: int) -> dict:
    """技術摘要：漲跌幅/均線/RSI/KD/MACD/量能，並以 0.618(下)/0.382(上) 做支撐/壓力提示。"""
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
    # 支撐/壓力（retracement 參考）
    high = float(tail["High"].max()); low = float(tail["Low"].min()); diff = high - low
    support = round(high - 0.618*diff, 2)
    resistance = round(high - 0.382*diff, 2)

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

# ===== 文字/新聞／AI 報告 =====
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

# --- Yahoo Finance 新聞（已修正：切片前轉字串） ---
@st.cache_data(show_spinner=False, ttl=600)
def fetch_news_yahoo(symbol: str, max_items: int = 3):
    """
    從 yfinance 取新聞。yfinance 某些欄位（如 summary/content）可能是 dict/list，
    在做切片前一律轉成字串避免 KeyError: slice(None, 1200, None)。
    """
    try:
        raw = yf.Ticker(symbol).news or []
    except Exception:
        raw = []

    items = []
    for n in raw[:max_items]:
        # 可能是 str / dict / list / None → 統一轉成字串
        raw_summary = n.get("summary") if isinstance(n, dict) else ""
        raw_content = n.get("content") if isinstance(n, dict) else ""
        text_sum = to_text(raw_summary) or to_text(raw_content) or ""
        text_sum = text_sum[:NEWS_ITEM_MAX_CHARS]  # 現在一定是字串，安全切片

        items.append({
            "title": (n.get("title") or "") if isinstance(n, dict) else "",
            "link": (n.get("link") or n.get("url") or "") if isinstance(n, dict) else "",
            "publisher": (n.get("publisher") or "Yahoo Finance") if isinstance(n, dict) else "Yahoo Finance",
            "published_ts": int(n.get("providerPublishTime", 0)) if isinstance(n, dict) and n.get("providerPublishTime") else None,
            "published_at_fmt": _epoch_to_local_str(n.get("providerPublishTime")) if isinstance(n, dict) and n.get("providerPublishTime") else "",
            "content": text_sum,
            "source": "Yahoo"
        })
    return items

# --- Google News RSS（可指定站台） ---
def _parse_google_news_rss(query: str, site: str | None, max_items: int = 3):
    q = quote_plus(query + (f" site:{site}" if site else ""))
    url = f"https://news.google.com/rss/search?q={q}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
    except Exception:
        return []
    try:
        root = ET.fromstring(r.text)
    except Exception:
        return []
    ns = {}
    items = []
    for item in root.findall(".//item", ns)[:max_items]:
        title = (item.findtext("title") or "").strip()
        link  = (item.findtext("link") or "").strip()
        pub   = (item.findtext("pubDate") or "").strip()
        src   = ""
        src_el = item.find("{*}source")
        if src_el is not None and src_el.text:
            src = src_el.text.strip()
        items.append({
            "title": title, "link": link,
            "publisher": src or (site or "Google News"),
            "published_ts": None, "published_at_fmt": _fmt_dt_str(pub),
            "content": "", "source": site or "GoogleNews"
        })
    return items

@st.cache_data(show_spinner=False, ttl=600)
def fetch_news_multi(symbol: str, display_name: str, each: int = 3):
    """三個網站：Yahoo + GoogleNews(UDN) + GoogleNews(CNYES)，各取 3 則"""
    q = display_name or symbol
    lst = []
    lst += fetch_news_yahoo(symbol, max_items=each)
    lst += _parse_google_news_rss(q, site="money.udn.com", max_items=each)
    lst += _parse_google_news_rss(q, site="news.cnyes.com", max_items=each)
    for it in lst:
        if not it.get("content"):
            it["content"] = it.get("title","")
    return lst[: (each*3)]

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
    except Exception:
        pass
    # fallback：若 LLM 失敗則以中性填回
    return [{
        "title": it.get("title",""), "published_at": it.get("published_at_fmt",""), "link": it.get("link",""),
        "summary": (it.get("content","") or "")[:200],
        "stock_sentiment": {"label":"Neutral","score":0.5},
        "article_sentiment": {"label":"Neutral","score":0.5},
        "relevance": 0.5
    } for it in (news_items or [])]

def build_overall_report(symbol: str, features: dict, results: list, temperature: float):
    """產生完整 Markdown 報告（含：市場情緒總結 / AI 分析結果）"""
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

    counts = {"Bullish":0,"Neutral":0,"Bearish":0}; rel = 0.0
    for r in (results or []):
        lab = r.get("stock_sentiment",{}).get("label","Neutral")
        counts[lab] = counts.get(lab,0) + 1
        rel += float(r.get("relevance",0.0) or 0.0)
    total = len(results); avg_rel = round(rel/total, 3) if total else 0.0

    payload = {
        "symbol": symbol,
        "features": features,
        "sentiment_stats": {
            "total": total,
            "bullish": counts["Bullish"],
            "neutral": counts["Neutral"],
            "bearish": counts["Bearish"],
            "avg_relevance": avg_rel
        },
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
        base = (
            f"## 市場情緒總結\n"
            f"- 近 {features['period_bars']} 根漲跌幅：{features['pct_change_period']}%\n"
            f"- 均線：SMA20 與 SMA60 為「{features['ma_trend']}」結構；"
            f"黃金交叉：{features['golden_cross']}；死亡交叉：{features['dead_cross']}\n"
            f"- RSI14：{features['rsi14']}（{features['rsi_state']}）；量能是否放大："
            f"{'是' if features['volume_spike'] else '否'}（近 30 日均量約 {features['volume_mean_period']:,}）\n"
            f"- 支撐區：{features['support_near']}；壓力區：{features['resistance_near']}\n\n"
            f"## AI 分析結果\n"
            f"### 分析總結\n- 資料不足以生成完整 AI 報告（{e}）。\n"
        )
        return base

def split_ai_report(text: str) -> tuple[str, str]:
    """把完整 Markdown 拆成『市場情緒總結』與『AI 分析結果』兩段，便於分區顯示。"""
    if not text: return "", ""
    m = re.search(r"^##\s*AI\s*分析結果.*$", text, flags=re.M)
    if not m:
        return text.strip(), ""
    part1 = text[:m.start()]
    part2 = text[m.start():]
    part1 = re.sub(r"^##\s*[^\n]+\n?", "", part1.strip(), count=1, flags=re.M)
    part2 = re.sub(r"^##\s*[^\n]+\n?", "", part2.strip(), count=1, flags=re.M)
    return part1.strip(), part2.strip()

def make_report_download(name: str, text: str):
    st.download_button("⬇️ 下載 Markdown 報告", data=io.BytesIO(text.encode("utf-8")), file_name=name, mime="text/markdown")

# ====== Sidebar：精簡控制（新聞固定開啟） ======
with st.sidebar:
    st.markdown("**基本設定**")
    st.caption("市場固定：台股（.TW / .TWO）")
    query = st.text_input("輸入名稱或代碼（支援模糊）", value="2330")

    yahoo_found = yahoo_search_tw(query, 20)
    numeric_candidates = _resolve_numeric_code_candidates(query)

    direct = []
    if re.fullmatch(r"\d{1,6}\.(?:TW|TWO)", query.strip().upper()):
        sym = query.strip().upper()
        if _verify_symbol_has_data(sym):
            try:
                info = yf.Ticker(sym).info or {}
                nm = info.get("longName") or info.get("shortName") or ""
            except Exception:
                nm = ""
            direct.append({"symbol": sym, "name": nm})

    merged = []
    seen = set()
    for seq in (direct, numeric_candidates, yahoo_found):
        for it in seq:
            if it["symbol"] not in seen:
                merged.append(it); seen.add(it["symbol"])

    if not merged and re.fullmatch(r"\d{4}", query.strip()):
        merged.append({"symbol": f"{query.strip()}.TW", "name": ""})

    options = [f'{it["symbol"]} — {it["name"]}' if it["name"] else it["symbol"] for it in merged] or ["（無結果，請輸入其他關鍵字）"]
    sel = st.selectbox("搜尋結果", options, index=0)
    if merged:
        sel_idx = options.index(sel); symbol = merged[sel_idx]["symbol"]; display_name = merged[sel_idx]["name"] or ""
    else:
        symbol, display_name = ("2330.TW", "")
    if not display_name: display_name = get_name_by_symbol(symbol)

    period   = DEFAULT_PERIOD
    interval = DEFAULT_INTERVAL
    lookback = DEFAULT_LOOKBACK

    st.markdown("---")
    st.markdown("**實驗功能：價格路徑預測**")
    predict_enabled = st.toggle("顯示未來一週預測", value=True)
    forecast_steps = 5 if interval == "1d" else 1

    st.markdown("---")
    st.caption(f"AI Temperature 已固定為 {FIXED_TEMPERATURE}（較詳細、較一致）")

    st.markdown("---")
    # 新聞固定開啟（移除切換鈕）
    st.caption("自動抓取新聞：Yahoo + UDN + 鉅亨（已固定啟用）")
    max_each_site = 3

    st.markdown("---")
    disclaimer = st.checkbox("我了解本工具僅供教育用途，非投資建議。", value=True)
    run_btn = st.button("產生分析", use_container_width=True)

# ====== 主流程 ======
def run_analysis(params: dict):
    symbol        = params["symbol"]
    display_name  = params.get("display_name") or get_name_by_symbol(symbol)
    period        = params["period"]
    interval      = params["interval"]
    lookback      = params["lookback"]
    temperature   = params["temperature"]
    max_each_site = params["max_each_site"]
    disclaimer    = params["disclaimer"]
    predict_enabled = params.get("predict_enabled", False)
    forecast_steps  = int(params.get("forecast_steps", 5))

    if not disclaimer:
        st.warning("請勾選『僅供教育用途』後再執行。"); st.stop()

    st.markdown(f"### {display_name}（ {symbol} ）" if display_name else f"### （ {symbol} ）")

    # === 資料 & 指標 ===
    data = fetch_ohlcv(symbol, period, interval)
    if data.empty: st.warning("抓不到資料，請確認代碼或期間設定。"); st.stop()
    df = compute_indicators(data)
    feats = summarize_features(df, lookback=lookback)

    # === 一、股價趨勢圖 ===
    st.markdown('<a id="sec-chart"></a>', unsafe_allow_html=True)
    st.markdown("## 一、股價趨勢圖")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="K",
        increasing=dict(line=dict(color="#26A69A", width=1.2), fillcolor="rgba(38,166,154,0.6)"),
        decreasing=dict(line=dict(color="#EF5350", width=1.2), fillcolor="rgba(239,83,80,0.6)")
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20", mode="lines",
                             line=dict(width=2.2, color="#60A5FA")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA60"], name="SMA60", mode="lines",
                             line=dict(width=2.2, color="#F9A8D4")), row=1, col=1)

    s20, s60 = df["SMA20"], df["SMA60"]
    cross_up_idx = (s20.shift(1) <= s60.shift(1)) & (s20 > s60)
    cross_dn_idx = (s20.shift(1) >= s60.shift(1)) & (s20 < s60)
    fig.add_trace(go.Scatter(x=df.index[cross_up_idx], y=df.loc[cross_up_idx, "Close"], mode="markers",
                             name="黃金交叉", marker_symbol="triangle-up", marker_size=10, marker_color="#22c55e"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index[cross_dn_idx], y=df.loc[cross_dn_idx, "Close"], mode="markers",
                             name="死亡交叉", marker_symbol="triangle-down", marker_size=10, marker_color="#f97316"), row=1, col=1)

    if feats["support_near"]:
        fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=feats["support_near"], y1=feats["support_near"],
                      xref="x", yref="y", line=dict(dash="dot", width=1.6, color="#10b981"), row=1, col=1)
    if feats["resistance_near"]:
        fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=feats["resistance_near"], y1=feats["resistance_near"],
                      xref="x", yref="y", line=dict(dash="dot", width=1.6, color="#ef4444"), row=1, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="rgba(100,116,139,0.7)"), row=2, col=1)

    if predict_enabled and forecast_steps > 0:
        try:
            base = df[["Open","High","Low","Close"]].dropna()
            fdf, med, p10, p90 = forecast_ohlc(base, steps=forecast_steps, interval=interval)
            purple = "#A855F7"
            fig.add_vrect(x0=df.index[-1], x1=fdf.index[-1], fillcolor="rgba(168,85,247,0.08)", line_width=0, row=1, col=1)
            fig.add_trace(go.Candlestick(
                x=fdf.index, open=fdf["Open"], high=fdf["High"], low=fdf["Low"], close=fdf["Close"], name="預測K",
                increasing=dict(line=dict(color=purple, width=1.2), fillcolor="rgba(168,85,247,0.45)"),
                decreasing=dict(line=dict(color=purple, width=1.2), fillcolor="rgba(168,85,247,0.45)"),
                opacity=0.85, whiskerwidth=0.3
            ), row=1, col=1)
            fig.add_trace(go.Scatter(x=med.index, y=p90, name="P90(收盤)", mode="lines",
                                     line=dict(width=1.2, dash="dashdot", color=purple)), row=1, col=1)
            fig.add_trace(go.Scatter(x=med.index, y=p10, name="P10(收盤)", mode="lines",
                                     line=dict(width=1.2, dash="dot", color=purple),
                                     fill="tonexty", fillcolor="rgba(168,85,247,0.12)"), row=1, col=1)
            fig.add_trace(go.Scatter(x=med.index, y=med, name="預測收盤(中位)", mode="lines+markers",
                                     line=dict(width=2.2, dash="dash", color=purple),
                                     marker=dict(size=5, symbol="diamond")), row=1, col=1)
            fig.update_xaxes(range=[df.index[0], fdf.index[-1] + pd.Timedelta(days=2)])
        except Exception as e:
            st.info(f"（預測層載入失敗：{e}）")

    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=620,
                      margin=dict(l=10, r=60, t=36, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # KPI
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最新收盤", feats["close_last"])
    c2.metric("近N根漲跌幅(%)", feats["pct_change_period"])
    c3.metric("支撐區(Fib 0.618)", _s(feats["support_near"]))
    c4.metric("壓力區(Fib 0.382)", _s(feats["resistance_near"]))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("RSI14", _s(feats["rsi14"]))
    c6.metric("量能是否放大", "是" if feats["volume_spike"] else "否",
              help=f"最新一根 vs 近 {VOLUME_LOOKBACK} 根均量；≥ {VOL_SPIKE_MULTIPLIER} 倍視為放大")
    c7.metric("量能基準(近30日)", f"{feats['volume_mean_period']:,}")
    c8.metric("最新量", f"{feats['volume_last']:,}")

    # === 技術輔助指標（KD / MACD） ===
    st.markdown('<a id="sec-tech"></a>', unsafe_allow_html=True)
    st.markdown("### 技術輔助指標")
    col_a, col_b = st.columns(2)
    fig_kd = go.Figure()
    fig_kd.add_trace(go.Scatter(x=df.index, y=df["KD_K"], mode="lines", name="%K(9)"))
    fig_kd.add_trace(go.Scatter(x=df.index, y=df["KD_D"], mode="lines", name="%D(3)"))
    fig_kd.add_hline(y=80, line_dash="dot"); fig_kd.add_hline(y=20, line_dash="dot")
    fig_kd.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
    with col_a: st.plotly_chart(fig_kd, use_container_width=True)

    fig_macd = go.Figure()
    fig_macd.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Hist"))
    fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines", name="MACD"))
    fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], mode="lines", name="Signal"))
    fig_macd.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=10), legend=dict(orientation="h"))
    with col_b: st.plotly_chart(fig_macd, use_container_width=True)

    # === 黃金切割率（僅表格） ===
    st.markdown("### 📐 黃金切割率點位（±0.382 / ±0.618，近 N 根幅度）")
    fib_tbl = compute_fib_posneg(df, lookback)
    if fib_tbl is not None and not fib_tbl.empty:
        st.dataframe(fib_tbl, use_container_width=True, hide_index=True)
    else:
        st.info("無法計算黃金切割率（資料不足）。")

    # === 二、情報新聞摘要（固定啟用） ===
    st.markdown('<a id="sec-news"></a>', unsafe_allow_html=True)
    st.markdown("## 二、情報新聞摘要")
    results = []
    items = []
    with st.spinner("抓取新聞中…"):
        items = fetch_news_multi(symbol, display_name, each=max_each_site)
    if items:
        with st.spinner("AI 情緒分析中…"):
            results = classify_news_with_gemini(symbol, items, FIXED_TEMPERATURE)

        def _agg(res):
            if not res: return {"total":0,"bullish":0,"neutral":0,"bearish":0,"bullish_ratio":0.0,"avg_rel":0.0}
            m = {"Bullish":0,"Neutral":0,"Bearish":0}
            s_rel = 0.0
            for r in res:
                m[r.get("stock_sentiment",{}).get("label","Neutral")] += 1
                s_rel += float(r.get("relevance",0.0) or 0.0)
            t = len(res)
            return {"total":t,"bullish":m["Bullish"],"neutral":m["Neutral"],"bearish":m["Bearish"],
                    "bullish_ratio": round(m["Bullish"]/t,3), "avg_rel": round(s_rel/t,3)}
        agg = _agg(results)

        r1, r2, r3, r4 = st.columns([2,1,1,1])
        with r2: st.metric("新聞總篇數", agg["total"])
        with r3: st.metric("正向比例", f'{int(agg["bullish_ratio"]*100)}%')
        with r4: st.metric("平均相關性", agg["avg_rel"])

        pos = int(agg["bullish_ratio"]*100)
        pie_df = pd.DataFrame({"label":["正向(Bullish)","中立/負向(others)"], "value":[pos, 100-pos]})
        st.plotly_chart(px.pie(pie_df, names="label", values="value", hole=0.55, title="情緒統計結果")
                        .update_layout(template="plotly_dark", margin=dict(l=0,r=0,t=40,b=0), height=300),
                        use_container_width=True)

        st.markdown("### 🧾 詳細新聞列表")
        for i, r in enumerate(results):
            meta = items[i] if i < len(items) else {}
            src = meta.get("publisher","")
            pub = meta.get("published_at_fmt","") or r.get("published_at","")
            stock_lab = r.get("stock_sentiment",{}).get("label","Neutral")
            art_lab   = r.get("article_sentiment",{}).get("label","Neutral")
            stock_badge = "badge-green" if stock_lab=="Bullish" else ("badge-red" if stock_lab=="Bearish" else "badge-gray")
            art_badge   = "badge-green" if art_lab=="Positive" else ("badge-red" if art_lab=="Negative" else "badge-gray")
            title = _s(r.get("title","(無標題)"))
            summary_txt = (_s(r.get('summary','')).strip())
            link_txt = _s(r.get('link','')).strip()
            link_html = f"<div style='margin-top:8px;'><a href=\"{link_txt}\" target=\"_blank\">查看原文</a></div>" if link_txt else ""
            meta_line = f"消息來源：{src or '—'}　|　發佈時間：{pub or '—'}　|　相關性分數：{_s(r.get('relevance',0))}"
            st.markdown(
                f"""
<div class="card">
  <div class="small" style="margin:6px 0;">{meta_line}</div>
  <div style="font-weight:600">{title}</div>
  <div style="margin:6px 0;">
    <span class="badge {stock_badge}">📈 股票情緒：{stock_lab}</span>
    <span class="badge {art_badge}">📰 文章情緒：{art_lab}</span>
  </div>
  <div style="white-space:pre-wrap; margin-top:6px;">{summary_txt}</div>
  {link_html}
</div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("目前抓不到新聞，已跳過情緒分析。")

    # === 三：AI 報告（僅顯示「AI 分析結果」章節） ===
    st.markdown('<a id="sec-ai"></a>', unsafe_allow_html=True)
    st.markdown("## 三、AI 分析結果")
    st.caption("※ 本工具為教育用途的分析輔助，不構成投資建議。")

    with st.spinner("AI 報告生成中…"):
        full_md = build_overall_report(symbol, feats, results, FIXED_TEMPERATURE)
        part_summary, part_ai = split_ai_report(full_md)

    # 只顯示 AI 章節；若模型未輸出拆段，則顯示完整內容作為保底
    if part_ai:
        st.markdown(part_ai)
    else:
        st.markdown(re.sub(r"^##\s*[^\n]+\n?", "", (full_md or "").strip(), count=1, flags=re.M))

    # 報告下載（Markdown）
    today = dt.datetime.now().strftime("%Y%m%d")
    report_name = f"{symbol.replace('.','_')}_report_{today}.md"
    make_report_download(report_name, full_md)

# ====== 事件處理 ======
if run_btn:
    st.session_state.last_params = {
        "symbol": symbol, "display_name": display_name,
        "period": DEFAULT_PERIOD, "interval": DEFAULT_INTERVAL, "lookback": DEFAULT_LOOKBACK,
        "temperature": FIXED_TEMPERATURE, "max_each_site": max_each_site,
        "disclaimer": True, "predict_enabled": predict_enabled, "forecast_steps": forecast_steps
    }
    run_analysis(st.session_state.last_params)
elif st.session_state.last_params:
    run_analysis(st.session_state.last_params)
