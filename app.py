import math
import time
import sqlite3
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

IST = ZoneInfo("Asia/Kolkata")

# =========================================================================
# PAGE CONFIGURATION
# =========================================================================
st.set_page_config(
    page_title="AI Trading Assistant V5 Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================================
# CUSTOM CSS DASHBOARD STYLING
# =========================================================================
st.markdown("""
<style>
    /* Dark Mode App Background */
    .stApp {
        background-color: #0E1117;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Card Container */
    .card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    
    .card-title {
        font-size: 12px;
        font-weight: 600;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    
    .card-value {
        font-size: 24px;
        font-weight: 700;
        color: #F0F6FC;
    }
    
    .card-sub {
        font-size: 12px;
        color: #8B949E;
        margin-top: 4px;
    }
    
    /* Signal Badges */
    .badge-call {
        background: rgba(0, 230, 118, 0.12);
        border: 1px solid #00E676;
        color: #00E676;
        padding: 10px 18px;
        border-radius: 10px;
        font-weight: 800;
        font-size: 24px;
        text-align: center;
        letter-spacing: 0.5px;
        box-shadow: 0 0 15px rgba(0, 230, 118, 0.2);
    }
    
    .badge-put {
        background: rgba(255, 82, 82, 0.12);
        border: 1px solid #FF5252;
        color: #FF5252;
        padding: 10px 18px;
        border-radius: 10px;
        font-weight: 800;
        font-size: 24px;
        text-align: center;
        letter-spacing: 0.5px;
        box-shadow: 0 0 15px rgba(255, 82, 82, 0.2);
    }
    
    .badge-neutral {
        background: rgba(255, 214, 0, 0.12);
        border: 1px solid #FFD600;
        color: #FFD600;
        padding: 10px 18px;
        border-radius: 10px;
        font-weight: 800;
        font-size: 20px;
        text-align: center;
        letter-spacing: 0.5px;
    }

    /* Alert Boxes */
    .status-warn {
        background: rgba(255, 214, 0, 0.08);
        border-left: 4px solid #FFD600;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 14px;
        color: #F0F6FC;
        font-size: 14px;
    }
    
    .status-danger {
        background: rgba(255, 82, 82, 0.08);
        border-left: 4px solid #FF5252;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 14px;
        color: #F0F6FC;
        font-size: 14px;
    }

    /* Checklist Badge */
    .check-pass {
        background: rgba(0, 230, 118, 0.06);
        border: 1px solid rgba(0, 230, 118, 0.3);
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 13.5px;
        color: #E6EDF3;
    }
    
    .check-fail {
        background: rgba(255, 82, 82, 0.06);
        border: 1px solid rgba(255, 82, 82, 0.3);
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 13.5px;
        color: #E6EDF3;
    }
    
    /* Streamlit Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 8px 16px;
        color: #8B949E;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1F242C !important;
        border-color: #58A6FF !important;
        color: #58A6FF !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================================
# DEFAULT LOT SIZES
# =========================================================================
DEFAULT_LOT_SIZES = {
    "NIFTY": 65,
    "NIFTY50": 65,
    "BANKNIFTY": 30,
    "FINNIFTY": 60,
    "MIDCPNIFTY": 120,
    "SENSEX": 20,
}

NSE_INDEX_SYMBOLS = {"NIFTY", "NIFTY50", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"}

# --- 1. SQLITE TRADE JOURNAL DATABASE ---
DB_NAME = "trade_journal.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            price REAL,
            bias TEXT,
            strike TEXT,
            entry REAL,
            sl REAL,
            target REAL,
            score INTEGER,
            pcr REAL,
            vix REAL,
            notes TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_trade(symbol, price, bias, strike, entry, sl, target, score, pcr, vix, notes="Auto-Logged"):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        INSERT INTO journal (timestamp, symbol, price, bias, strike, entry, sl, target, score, pcr, vix, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S'), symbol, price, bias, strike,
          entry, sl, target, score, pcr, vix, notes))
    conn.commit()
    conn.close()

def get_journal_data():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM journal ORDER BY id DESC", conn)
    conn.close()
    return df

init_db()

# --- 2. TELEGRAM ALERT DISPATCHER ---
def send_telegram_alert(message: str) -> bool:
    bot_token = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = st.secrets.get("TELEGRAM_CHAT_ID", "")
    if not bot_token or not chat_id:
        st.error("Telegram credentials missing in Streamlit Secrets!")
        return False
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        res = requests.post(url, json=payload, timeout=10)
        return res.json().get("ok", False)
    except Exception as e:
        st.error(f"Telegram Delivery Error: {e}")
        return False

# =========================================================================
# 3. ROBUST YAHOO HTTP FETCH WITH CLOUD IP BYPASS
# =========================================================================
def fetch_yahoo_candles(ticker_sym: str, interval="5m", range_pd="5d") -> pd.DataFrame:
    for base_url in ["https://query2.finance.yahoo.com", "https://query1.finance.yahoo.com"]:
        try:
            url = f"{base_url}/v8/finance/chart/{ticker_sym}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Accept": "*/*"
            }
            params = {"range": range_pd, "interval": interval}
            r = requests.get(url, headers=headers, params=params, timeout=6)
            if r.status_code == 200:
                result = r.json().get("chart", {}).get("result", [])
                if result:
                    timestamps = result[0].get("timestamp", [])
                    quote = result[0].get("indicators", {}).get("quote", [{}])[0]
                    if timestamps and quote and quote.get("close"):
                        df = pd.DataFrame({
                            "Open": quote.get("open"),
                            "High": quote.get("high"),
                            "Low": quote.get("low"),
                            "Close": quote.get("close"),
                            "Volume": quote.get("volume", [1000]*len(timestamps))
                        }, index=pd.to_datetime(timestamps, unit='s', utc=True))
                        df["Volume"] = df["Volume"].fillna(1000)
                        df = df.dropna(subset=["Close"])
                        if not df.empty:
                            return df
        except Exception:
            continue
    return pd.DataFrame()

def generate_fallback_candles(spot_price: float) -> pd.DataFrame:
    now = datetime.now(IST)
    dates = [now - timedelta(minutes=5 * i) for i in range(100, 0, -1)]
    np.random.seed(42)
    volatility = spot_price * 0.0015
    closes = spot_price + np.cumsum(np.random.randn(100) * volatility)
    closes[-1] = spot_price
    
    df = pd.DataFrame({
        "Open": closes - (np.random.rand(100) * volatility * 0.5),
        "High": closes + (np.random.rand(100) * volatility),
        "Low": closes - (np.random.rand(100) * volatility),
        "Close": closes,
        "Volume": np.random.randint(5000, 20000, size=100)
    }, index=pd.to_datetime(dates, utc=True))
    return df

# =========================================================================
# 4. MARKET HOURS / SESSION HELPERS
# =========================================================================
def market_status():
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False, "Market Closed (Weekend)"
    open_t = now.replace(hour=9, minute=15, second=0, microsecond=0)
    close_t = now.replace(hour=15, minute=30, second=0, microsecond=0)
    if now < open_t:
        return False, "Market Closed (Pre-Open) - showing last session data"
    if now > close_t:
        return False, "Market Closed - showing last session data"
    return True, "Market Open"

# =========================================================================
# 5. TECHNICAL INDICATORS & SMC CALCULATIONS
# =========================================================================
def calculate_session_vwap(df: pd.DataFrame) -> pd.Series:
    idx = df.index
    session_date = idx.tz_convert(IST).date if idx.tz is not None else idx.date
    df = df.copy()
    df["_session"] = session_date
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    tpv = typical_price * df['Volume']
    vwap = tpv.groupby(df["_session"]).cumsum() / df['Volume'].groupby(df["_session"]).cumsum()
    return vwap

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['High'], df['Low'], df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_supertrend(df: pd.DataFrame, period: int = 7, multiplier: float = 3.0) -> pd.Series:
    hl2 = (df['High'] + df['Low']) / 2
    atr = calculate_atr(df, period)
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    
    supertrend = pd.Series(index=df.index, dtype=float)
    in_uptrend = True
    
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > upperband.iloc[i-1]:
            in_uptrend = True
        elif df['Close'].iloc[i] < lowerband.iloc[i-1]:
            in_uptrend = False
            
        supertrend.iloc[i] = lowerband.iloc[i] if in_uptrend else upperband.iloc[i]
        
    return supertrend

def detect_fvg(df: pd.DataFrame):
    if len(df) < 3:
        return False, "N/A"
    c1_high = df['High'].iloc[-3]
    c3_low = df['Low'].iloc[-1]
    c1_low = df['Low'].iloc[-3]
    c3_high = df['High'].iloc[-1]
    
    if c3_low > c1_high:
        return True, "Bullish FVG Active"
    elif c3_high < c1_low:
        return True, "Bearish FVG Active"
    return False, "No Imbalance"

def opening_range(df_today: pd.DataFrame, minutes: int = 15):
    if df_today.empty:
        return None, None
    idx = df_today.index
    session_date = idx.tz_convert(IST).date if idx.tz is not None else idx.date
    day_open = datetime.combine(session_date[0], datetime.min.time(), tzinfo=IST).replace(hour=9, minute=15)
    day_open_end = day_open + timedelta(minutes=minutes)
    mask = (idx.tz_convert(IST) >= day_open) & (idx.tz_convert(IST) < day_open_end) if idx.tz is not None else (idx >= day_open.replace(tzinfo=None)) & (idx < day_open_end.replace(tzinfo=None))
    orb_slice = df_today[mask]
    if orb_slice.empty:
        return None, None
    return orb_slice['High'].max(), orb_slice['Low'].min()

# =========================================================================
# 6. INDIA VIX
# =========================================================================
@st.cache_data(ttl=60)
def get_india_vix():
    try:
        vix_df = fetch_yahoo_candles("^INDIAVIX", interval="15m", range_pd="5d")
        if vix_df.empty:
            return 13.5, "NORMAL (Fallback)"
        vix_val = round(float(vix_df['Close'].iloc[-1]), 2)
        if vix_val < 12:
            regime = "LOW (range-bound risk)"
        elif vix_val < 18:
            regime = "NORMAL"
        elif vix_val < 22:
            regime = "ELEVATED (caution)"
        else:
            regime = "HIGH (avoid naked option buying)"
        return vix_val, regime
    except Exception:
        return 13.5, "NORMAL (Fallback)"

# =========================================================================
# 7a. GROWW TRADE API - OFFICIAL OPTION CHAIN
# =========================================================================
GROWW_INSTRUMENTS_URL = "https://growwapi-assets.groww.in/instruments/instrument.csv"

def get_groww_headers():
    token = st.secrets.get("GROWW_ACCESS_TOKEN", "")
    if not token:
        return None
    return {"Accept": "application/json", "X-API-VERSION": "1.0", "Authorization": f"Bearer {token}"}

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_groww_instruments():
    try:
        from io import StringIO
        r = requests.get(GROWW_INSTRUMENTS_URL, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        return df
    except Exception:
        return None

def get_expiry_and_lotsize(raw_sym: str, instruments_df):
    if instruments_df is None or instruments_df.empty:
        return None, None
    try:
        sub = instruments_df[
            (instruments_df["underlying_symbol"] == raw_sym) &
            (instruments_df["segment"] == "FNO")
        ].copy()
        if sub.empty:
            return None, None
        sub["expiry_date"] = pd.to_datetime(sub["expiry_date"], errors="coerce")
        sub = sub.dropna(subset=["expiry_date"])
        today = pd.Timestamp(datetime.now(IST).date())
        future = sub[sub["expiry_date"] >= today]
        if future.empty:
            return None, None
        nearest = future["expiry_date"].min()
        lot_row = future[future["expiry_date"] == nearest]
        lot_size = int(lot_row["lot_size"].iloc[0]) if not lot_row.empty else None
        return nearest.strftime("%Y-%m-%d"), lot_size
    except Exception:
        return None, None

@st.cache_data(ttl=15, show_spinner=False)
def fetch_groww_option_chain(raw_sym: str, exchange: str, expiry_date: str):
    headers = get_groww_headers()
    if not headers or not expiry_date:
        return None
    try:
        url = f"https://api.groww.in/v1/option-chain/exchange/{exchange}/underlying/{raw_sym}"
        r = requests.get(url, headers=headers, params={"expiry_date": expiry_date}, timeout=8)
        if r.status_code != 200:
            return None
        payload = r.json().get("payload", {})
        underlying = payload.get("underlying_ltp")
        strikes_raw = payload.get("strikes", {})
        if not underlying or not strikes_raw:
            return None

        total_call_oi, total_put_oi = 0, 0
        strikes = []
        for strike_str, legs in strikes_raw.items():
            try:
                strike = float(strike_str)
            except (TypeError, ValueError):
                continue
            ce = legs.get("CE") or {}
            pe = legs.get("PE") or {}
            ce_oi = ce.get("open_interest") or 0
            pe_oi = pe.get("open_interest") or 0
            total_call_oi += ce_oi
            total_put_oi += pe_oi
            strikes.append({
                "strike": strike,
                "CE": {"lastPrice": ce.get("ltp"), "openInterest": ce_oi,
                       "impliedVolatility": (ce.get("greeks") or {}).get("iv"),
                       "delta": (ce.get("greeks") or {}).get("delta"),
                       "theta": (ce.get("greeks") or {}).get("theta")},
                "PE": {"lastPrice": pe.get("ltp"), "openInterest": pe_oi,
                       "impliedVolatility": (pe.get("greeks") or {}).get("iv"),
                       "delta": (pe.get("greeks") or {}).get("delta"),
                       "theta": (pe.get("greeks") or {}).get("theta")},
            })
        if not strikes:
            return None

        pcr = round(total_put_oi / total_call_oi, 2) if total_call_oi > 0 else None
        atm_row = min(strikes, key=lambda x: abs(x["strike"] - underlying))

        max_pain_strike, min_loss = None, None
        for s in strikes:
            loss = 0
            for s2 in strikes:
                loss += max(0, s["strike"] - s2["strike"]) * s2["CE"]["openInterest"]
                loss += max(0, s2["strike"] - s["strike"]) * s2["PE"]["openInterest"]
            if min_loss is None or loss < min_loss:
                min_loss, max_pain_strike = loss, s["strike"]

        return {
            "underlying": underlying, "expiry": expiry_date, "pcr": pcr,
            "total_call_oi": total_call_oi, "total_put_oi": total_put_oi,
            "atm_strike": atm_row["strike"], "atm_ce": atm_row["CE"], "atm_pe": atm_row["PE"],
            "max_pain": max_pain_strike, "source": "Groww (live)",
        }
    except Exception:
        return None

# =========================================================================
# 7b. NSE SCRAPER - FALLBACK ONLY
# =========================================================================
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "*/*",
    "Referer": "https://www.nseindia.com/option-chain",
}

@st.cache_data(ttl=45, show_spinner=False)
def fetch_nse_option_chain(symbol: str, is_index: bool):
    try:
        session = requests.Session()
        session.headers.update(NSE_HEADERS)
        session.get("https://www.nseindia.com", timeout=6)
        session.get("https://www.nseindia.com/option-chain", timeout=6)

        url = f"https://www.nseindia.com/api/option-chain-{'indices' if is_index else 'equities'}?symbol={symbol}"
        resp = session.get(url, timeout=8)
        if resp.status_code != 200:
            return None
        data = resp.json()

        records = data.get("records", {})
        underlying = records.get("underlyingValue")
        expiries = records.get("expiryDates", [])
        if not underlying or not expiries:
            return None
        nearest_expiry = expiries[0]

        rows = [r for r in records.get("data", []) if r.get("expiryDate") == nearest_expiry]
        if not rows:
            return None

        total_call_oi, total_put_oi = 0, 0
        strikes = []
        for r in rows:
            ce, pe, strike = r.get("CE"), r.get("PE"), r.get("strikePrice")
            if ce:
                total_call_oi += ce.get("openInterest", 0)
            if pe:
                total_put_oi += pe.get("openInterest", 0)
            strikes.append({"strike": strike, "CE": ce, "PE": pe})

        pcr = round(total_put_oi / total_call_oi, 2) if total_call_oi > 0 else None
        atm_row = min(strikes, key=lambda x: abs(x["strike"] - underlying))

        max_pain_strike, min_loss = None, None
        for s in strikes:
            loss = 0
            for s2 in strikes:
                if s2["CE"]:
                    loss += max(0, s["strike"] - s2["strike"]) * s2["CE"].get("openInterest", 0)
                if s2["PE"]:
                    loss += max(0, s2["strike"] - s["strike"]) * s2["PE"].get("openInterest", 0)
            if min_loss is None or loss < min_loss:
                min_loss, max_pain_strike = loss, s["strike"]

        return {
            "underlying": underlying, "expiry": nearest_expiry, "pcr": pcr,
            "total_call_oi": total_call_oi, "total_put_oi": total_put_oi,
            "atm_strike": atm_row["strike"], "atm_ce": atm_row["CE"], "atm_pe": atm_row["PE"],
            "max_pain": max_pain_strike, "source": "NSE live",
        }
    except Exception:
        return None

def parse_expiry_date(expiry_str: str):
    for fmt in ("%Y-%m-%d", "%d-%b-%Y"):
        try:
            return datetime.strptime(expiry_str, fmt).date()
        except (ValueError, TypeError):
            continue
    return None

def is_expiry_today(expiry_str: str) -> bool:
    try:
        expiry_date = parse_expiry_date(expiry_str)
        return expiry_date == datetime.now(IST).date() if expiry_date else False
    except Exception:
        return False

# =========================================================================
# 8. CORE ANALYSIS ENGINE
# =========================================================================
def get_atm_display_strike(symbol: str, spot_price: float, bias: str, nse_chain) -> str:
    if nse_chain and nse_chain.get("atm_strike"):
        strike = nse_chain["atm_strike"]
    else:
        step = 100 if ("BANKNIFTY" in symbol or "SENSEX" in symbol) else (50 if "NIFTY" in symbol else 10)
        strike = round(spot_price / step) * step
    option_type = "CE" if bias == "BUY CALL" else "PE"
    return f"{int(strike)} {option_type}"

@st.cache_data(ttl=15)
def analyze_market(symbol: str, vix_val, orb_confirm_pct: float):
    raw_sym = symbol.upper().strip()
    is_index = raw_sym in NSE_INDEX_SYMBOLS

    index_map = {"NIFTY": "^NSEI", "NIFTY50": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN"}
    ticker_sym = index_map.get(raw_sym, raw_sym)
    if not ticker_sym.startswith("^") and not ticker_sym.endswith(".NS"):
        ticker_sym += ".NS"

    instruments_df = fetch_groww_instruments()
    groww_expiry, groww_lot_size = get_expiry_and_lotsize(raw_sym, instruments_df)
    groww_exchange = "BSE" if raw_sym == "SENSEX" else "NSE"
    nse_chain = fetch_groww_option_chain(raw_sym, groww_exchange, groww_expiry)
    if nse_chain is None:
        nse_chain = fetch_nse_option_chain(raw_sym, is_index)
    if nse_chain is not None and groww_lot_size:
        nse_chain["lot_size"] = groww_lot_size

    df_5m = fetch_yahoo_candles(ticker_sym, interval="5m", range_pd="5d")
    if df_5m.empty or len(df_5m) < 5:
        df_5m = fetch_yahoo_candles(ticker_sym, interval="15m", range_pd="7d")

    if df_5m.empty:
        live_spot = nse_chain.get("underlying") if nse_chain else (24500.0 if "NIFTY" in raw_sym else 52000.0)
        df_5m = generate_fallback_candles(live_spot)

    df_15m = fetch_yahoo_candles(ticker_sym, interval="15m", range_pd="10d")
    df_1h = fetch_yahoo_candles(ticker_sym, interval="1h", range_pd="1mo")
    df_1d = fetch_yahoo_candles(ticker_sym, interval="1d", range_pd="6mo")

    if df_15m.empty: df_15m = df_5m
    if df_1h.empty: df_1h = df_5m
    if df_1d.empty: df_1d = df_5m

    current_price = df_5m['Close'].iloc[-1]
    prev_close = df_1d['Close'].iloc[-2] if len(df_1d) > 1 else current_price
    day_high = df_5m['High'].max()
    day_low = df_5m['Low'].min()
    price_change_pct = ((current_price - prev_close) / prev_close) * 100

    trend_1d = "BULLISH" if len(df_1d) >= 20 and df_1d['Close'].iloc[-1] > df_1d['Close'].rolling(20).mean().iloc[-1] else ("BULLISH" if df_1d['Close'].iloc[-1] >= prev_close else "BEARISH")
    trend_1h = "BULLISH" if len(df_1h) >= 20 and df_1h['Close'].iloc[-1] > df_1h['Close'].rolling(20).mean().iloc[-1] else ("BULLISH" if df_1h['Close'].iloc[-1] >= prev_close else "BEARISH")
    trend_15m = "BULLISH" if len(df_15m) >= 20 and df_15m['Close'].iloc[-1] > df_15m['Close'].rolling(20).mean().iloc[-1] else ("BULLISH" if df_15m['Close'].iloc[-1] >= prev_close else "BEARISH")
    trend_5m = "BULLISH" if len(df_5m) >= 20 and df_5m['Close'].iloc[-1] > df_5m['Close'].rolling(20).mean().iloc[-1] else ("BULLISH" if df_5m['Close'].iloc[-1] >= prev_close else "BEARISH")

    df_5m['VWAP'] = calculate_session_vwap(df_5m)
    current_vwap = df_5m['VWAP'].iloc[-1] if not df_5m['VWAP'].dropna().empty else current_price
    above_vwap = current_price >= current_vwap

    df_5m['ATR'] = calculate_atr(df_5m, 14)
    current_atr = df_5m['ATR'].iloc[-1] if not df_5m['ATR'].dropna().empty else current_price * 0.003
    if np.isnan(current_atr) or current_atr <= 0:
        current_atr = current_price * 0.003

    df_5m['RSI'] = calculate_rsi(df_5m, 14)
    current_rsi = df_5m['RSI'].iloc[-1] if not df_5m['RSI'].dropna().empty else 50.0

    df_5m['Supertrend'] = calculate_supertrend(df_5m, 7, 3.0)
    supertrend_val = df_5m['Supertrend'].iloc[-1] if not df_5m['Supertrend'].dropna().empty else current_price
    above_supertrend = current_price >= supertrend_val

    fvg_active, fvg_desc = detect_fvg(df_5m)

    idx = df_5m.index
    today_key = idx.tz_convert(IST).date if idx.tz is not None else idx.date
    last_day = today_key[-1]
    try:
        df_today = df_5m[[d == last_day for d in today_key]]
    except Exception:
        df_today = df_5m.tail(75)

    orb_high, orb_low = opening_range(df_today, minutes=15)
    orb_bull = orb_high is not None and current_price > orb_high
    orb_bear = orb_low is not None and current_price < orb_low

    pcr = nse_chain["pcr"] if nse_chain and nse_chain["pcr"] is not None else None
    expiry_today = is_expiry_today(nse_chain["expiry"]) if nse_chain else False

    # --- CONFIDENCE SCORE CALCULATION ---
    score = 0
    checks = {}

    aligned_bullish = (trend_1d == "BULLISH" and trend_1h == "BULLISH" and trend_15m == "BULLISH")
    aligned_bearish = (trend_1d == "BEARISH" and trend_1h == "BEARISH" and trend_15m == "BEARISH")

    if aligned_bullish or aligned_bearish:
        score += 20
        checks["MTF Alignment"] = (True, "1D, 1H, 15M Aligned")
    else:
        checks["MTF Alignment"] = (False, "Timeframe Conflict")

    if (aligned_bullish and above_vwap) or (aligned_bearish and not above_vwap):
        score += 20
        checks["Session VWAP"] = (True, f"{'Above' if above_vwap else 'Below'} session VWAP")
    else:
        checks["Session VWAP"] = (False, "Conflicting Price vs Session VWAP")

    if (aligned_bullish and above_supertrend) or (aligned_bearish and not above_supertrend):
        score += 15
        checks["Supertrend (7,3)"] = (True, f"{'Bullish' if above_supertrend else 'Bearish'} Supertrend")
    else:
        checks["Supertrend (7,3)"] = (False, "Price on wrong side of Supertrend")

    if (aligned_bullish and current_rsi >= 50 and current_rsi <= 70) or \
       (aligned_bearish and current_rsi <= 50 and current_rsi >= 30):
        score += 10
        checks["RSI Momentum"] = (True, f"RSI = {current_rsi:.1f} (In trend zone)")
    else:
        checks["RSI Momentum"] = (False, f"RSI = {current_rsi:.1f} (Overbought/Oversold/Flat)")

    if (aligned_bullish and orb_bull) or (aligned_bearish and orb_bear):
        score += 15
        checks["Opening Range Breakout"] = (True, "Price beyond opening range in trend direction")
    else:
        checks["Opening Range Breakout"] = (False, "No confirmed ORB breakout")

    if pcr is None:
        checks["Option Chain (PCR)"] = (False, "Option data unavailable - not scored")
    elif (aligned_bullish and pcr >= 1.0) or (aligned_bearish and pcr < 1.0):
        score += 20
        checks["Option Chain (PCR)"] = (True, f"PCR = {pcr}")
    else:
        checks["Option Chain (PCR)"] = (False, f"PCR = {pcr} (Divergence)")

    vix_block = vix_val is not None and vix_val >= 22
    now_ist = datetime.now(IST)
    expiry_cutoff = expiry_today and now_ist.hour >= 14 and now_ist.minute >= 30

    bias, strike, sl, target1, target2 = "NEUTRAL / NO TRADE", "N/A", 0.0, 0.0, 0.0
    rrr, breakeven = "N/A", 0.0
    premium_info = None

    if score >= 70 and aligned_bullish and above_vwap and above_supertrend and not vix_block and not expiry_cutoff:
        bias = "BUY CALL"
    elif score >= 70 and aligned_bearish and not above_vwap and not above_supertrend and not vix_block and not expiry_cutoff:
        bias = "BUY PUT"

    if bias != "NEUTRAL / NO TRADE":
        strike = get_atm_display_strike(raw_sym, current_price, bias, nse_chain)
        atr_mult_sl, atr_mult_t1, atr_mult_t2 = 1.0, 1.5, 2.5
        if bias == "BUY CALL":
            sl = round(current_price - atr_mult_sl * current_atr, 2)
            target1 = round(current_price + atr_mult_t1 * current_atr, 2)
            target2 = round(current_price + atr_mult_t2 * current_atr, 2)
        else:
            sl = round(current_price + atr_mult_sl * current_atr, 2)
            target1 = round(current_price - atr_mult_t1 * current_atr, 2)
            target2 = round(current_price - atr_mult_t2 * current_atr, 2)

        risk_dist = abs(current_price - sl)
        reward_dist = abs(target1 - current_price)
        rrr = f"1 : {round(reward_dist / risk_dist, 2)}" if risk_dist > 0 else "N/A"

        leg = nse_chain["atm_ce"] if (nse_chain and bias == "BUY CALL") else (nse_chain["atm_pe"] if nse_chain else None)
        data_source = nse_chain.get("source", "NSE live") if nse_chain else "Delta Scaled (Spot Move)"
        
        if leg and leg.get("lastPrice"):
            prem = float(leg.get("lastPrice"))
            iv_val = leg.get("impliedVolatility")
            oi_val = leg.get("openInterest")
            chg_oi_val = leg.get("changeinOpenInterest")
            delta_val = leg.get("delta")
            theta_val = leg.get("theta")
        else:
            strike_num = float(strike.split()[0]) if strike != "N/A" else current_price
            dist_from_strike = abs(current_price - strike_num)
            prem = round(dist_from_strike + (current_atr * 3.5), 2)
            iv_val, oi_val, chg_oi_val = 14.5, "N/A", "N/A"
            delta_val, theta_val = (0.50 if bias == "BUY CALL" else -0.50), round(-prem * 0.08, 2)

        premium_info = {
            "source": data_source, "premium": prem, "iv": iv_val,
            "oi": oi_val, "chg_oi": chg_oi_val,
            "delta": delta_val, "theta_per_day": theta_val
        }
        
        strike_num = float(strike.split()[0]) if strike != "N/A" else current_price
        breakeven = round(strike_num + prem, 2) if bias == "BUY CALL" else round(strike_num - prem, 2)

    return {
        "symbol": raw_sym, "price": current_price, "change_pct": price_change_pct,
        "high": day_high, "low": day_low, "vwap": current_vwap, "atr": current_atr,
        "rsi": current_rsi, "supertrend": supertrend_val, "fvg_active": fvg_active, "fvg_desc": fvg_desc,
        "pcr": pcr, "bias": bias, "strike": strike, "sl": sl, "target1": target1, "target2": target2,
        "rrr": rrr, "breakeven": breakeven, "score": score, "checks": checks,
        "trends": {"1D": trend_1d, "1H": trend_1h, "15M": trend_15m, "5M": trend_5m},
        "orb_high": orb_high, "orb_low": orb_low,
        "nse_chain": nse_chain, "expiry_today": expiry_today, "expiry_cutoff": expiry_cutoff,
        "vix_block": vix_block, "premium_info": premium_info, "df_5m": df_5m
    }

# =========================================================================
# PLOTLY CANDLESTICK CHART BUILDER
# =========================================================================
def render_candlestick_chart(df: pd.DataFrame, symbol: str, orb_high=None, orb_low=None):
    if df.empty:
        return None
    
    df_chart = df.tail(80).copy()
    
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=[0.75, 0.25]
    )

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df_chart.index,
        open=df_chart['Open'],
        high=df_chart['High'],
        low=df_chart['Low'],
        close=df_chart['Close'],
        name="Price",
        increasing_line_color="#00E676",
        decreasing_line_color="#FF5252"
    ), row=1, col=1)

    # VWAP Line
    if 'VWAP' in df_chart.columns:
        fig.add_trace(go.Scatter(
            x=df_chart.index, y=df_chart['VWAP'],
            mode='lines', name='Session VWAP',
            line=dict(color='#00E5FF', width=1.5, dash='dot')
        ), row=1, col=1)

    # Supertrend Line
    if 'Supertrend' in df_chart.columns:
        fig.add_trace(go.Scatter(
            x=df_chart.index, y=df_chart['Supertrend'],
            mode='lines', name='Supertrend',
            line=dict(color='#FF9100', width=1.5)
        ), row=1, col=1)

    # ORB High / Low
    if orb_high:
        fig.add_hline(y=orb_high, line_dash="dash", line_color="#00E676", annotation_text="ORB High", row=1, col=1)
    if orb_low:
        fig.add_hline(y=orb_low, line_dash="dash", line_color="#FF5252", annotation_text="ORB Low", row=1, col=1)

    # Volume Bars
    colors = ['#00E676' if c >= o else '#FF5252' for c, o in zip(df_chart['Close'], df_chart['Open'])]
    fig.add_trace(go.Bar(
        x=df_chart.index, y=df_chart['Volume'],
        name="Volume",
        marker_color=colors,
        opacity=0.6
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=450,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#161B22",
        plot_bgcolor="#161B22",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Inter, sans-serif", color="#8B949E")
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#21262D')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#21262D')
    
    return fig

# =========================================================================
# STREAMLIT DASHBOARD UI
# =========================================================================

# --- SIDEBAR CONFIGURATION ---
st.sidebar.markdown("### Control Panel")
user_symbol = st.sidebar.text_input("Symbol Ticker", value="NIFTY").upper().strip()

st.sidebar.markdown("---")
st.sidebar.markdown("### Auto-Refresh Engine")
enable_autorefresh = st.sidebar.checkbox("Enable Auto-Refresh", value=False)
refresh_interval = st.sidebar.slider("Interval (Seconds)", min_value=15, max_value=120, value=30)
if enable_autorefresh:
    st_autorefresh(interval=refresh_interval * 1000, key="market_auto_refresh")

st.sidebar.markdown("---")
st.sidebar.markdown("### Position Sizing")
_instruments_df = fetch_groww_instruments()
_live_expiry, _live_lot = get_expiry_and_lotsize(user_symbol, _instruments_df)
if _live_lot:
    st.sidebar.caption(f"Lot size fetched live: {_live_lot} (Expiry: {_live_expiry})")
    default_lot = _live_lot
else:
    st.sidebar.caption("Live lot size fallback active.")
    default_lot = DEFAULT_LOT_SIZES.get(user_symbol, 1)

capital = st.sidebar.number_input("Trading Capital (INR)", min_value=1000, value=100000, step=1000)
risk_pct = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 1.5, 0.5)
lot_size = st.sidebar.number_input("Lot Size", min_value=1, value=default_lot, step=1)
premium_sl_pct = st.sidebar.slider("Premium Stop-Loss (%)", 10, 50, 30, 5)

# --- HEADER SECTION ---
vix_val, vix_regime = get_india_vix()
is_open, status_label = market_status()

header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title("AI TRADING ASSISTANT PRO")
    st.caption("Institutional Market Screener | Session VWAP | Supertrend | RSI Momentum | SMC FVG | Option Chain")

with header_col2:
    st.markdown(f"""
    <div class="card" style="padding:12px; text-align:right;">
        <div class="card-title">Market Status</div>
        <div style="font-weight:700; color:{'#00E676' if is_open else '#FFD600'};">{status_label}</div>
        <div class="card-sub">India VIX: <b>{vix_val}</b> ({vix_regime})</div>
    </div>
    """, unsafe_allow_html=True)

if not is_open:
    st.markdown(f'<div class="status-warn"><b>Market Notice:</b> {status_label}</div>', unsafe_allow_html=True)

if not get_groww_headers():
    st.markdown(
        '<div class="status-warn"><b>Groww API Notice:</b> Add GROWW_ACCESS_TOKEN in Streamlit Secrets for live option chain metrics.</div>',
        unsafe_allow_html=True
    )

tab_screener, tab_journal = st.tabs(["Live Screener & Signal", "Trade Journal"])

with tab_screener:
    data = analyze_market(user_symbol, vix_val, orb_confirm_pct=0.0)

    if data:
        if data["expiry_today"]:
            st.markdown('<div class="status-warn"><b>Expiry Day Notice:</b> Option buying auto-restricted after 2:30 PM IST.</div>', unsafe_allow_html=True)
        if data["vix_block"]:
            st.markdown('<div class="status-danger"><b>High Volatility Warning:</b> India VIX is elevated (>=22). Fresh option buying paused.</div>', unsafe_allow_html=True)

        # --- HERO METRIC CARDS ---
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        with kpi1:
            delta_color = "#00E676" if data['change_pct'] >= 0 else "#FF5252"
            st.markdown(f"""
            <div class="card">
                <div class="card-title">{data['symbol']} Spot Price</div>
                <div class="card-value">INR {data['price']:,.2f}</div>
                <div class="card-sub" style="color:{delta_color}; font-weight:700;">{data['change_pct']:+.2f}% Day Change</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi2:
            bias_class = "badge-call" if data["bias"] == "BUY CALL" else ("badge-put" if data["bias"] == "BUY PUT" else "badge-neutral")
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <div class="card-title">AI Decision Bias</div>
                <div class="{bias_class}">{data["bias"]}</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi3:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">Confidence Score</div>
                <div class="card-value">{data['score']}%</div>
                <div class="card-sub">Quantitative Matrix Match</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi4:
            pcr_str = f"{data['pcr']}" if data['pcr'] is not None else "N/A"
            pcr_color = "#00E676" if data['pcr'] and data['pcr'] >= 1.0 else "#FF5252"
            st.markdown(f"""
            <div class="card">
                <div class="card-title">Put-Call Ratio (PCR)</div>
                <div class="card-value" style="color:{pcr_color};">{pcr_str}</div>
                <div class="card-sub">Sentiment Balance</div>
            </div>
            """, unsafe_allow_html=True)

        # --- TRADE EXECUTION PARAMETERS CARD ---
        if data["bias"] != "NEUTRAL / NO TRADE":
            st.markdown("### Trade Execution Matrix")
            p1, p2, p3, p4, p5 = st.columns(5)
            
            p1.markdown(f"""<div class="card"><div class="card-title">Strike Option</div><div class="card-value" style="font-size:20px; color:#58A6FF;">{data['strike']}</div></div>""", unsafe_allow_html=True)
            p2.markdown(f"""<div class="card"><div class="card-title">Spot Entry</div><div class="card-value" style="font-size:20px;">INR {data['price']:,.2f}</div></div>""", unsafe_allow_html=True)
            p3.markdown(f"""<div class="card"><div class="card-title">Spot Stop Loss</div><div class="card-value" style="font-size:20px; color:#FF5252;">INR {data['sl']:,.2f}</div></div>""", unsafe_allow_html=True)
            p4.markdown(f"""<div class="card"><div class="card-title">Targets (T1 / T2)</div><div class="card-value" style="font-size:18px; color:#00E676;">INR {data['target1']:,.0f} / {data['target2']:,.0f}</div></div>""", unsafe_allow_html=True)
            p5.markdown(f"""<div class="card"><div class="card-title">Risk-Reward</div><div class="card-value" style="font-size:20px;">{data['rrr']}</div></div>""", unsafe_allow_html=True)

            if data["premium_info"]:
                pi = data["premium_info"]
                st.markdown("##### Option Analytics & Position Sizing")
                g1, g2, g3, g4, g5 = st.columns(5)
                
                g1.markdown(f"""<div class="card"><div class="card-title">Entry Premium</div><div class="card-value" style="font-size:20px;">INR {pi['premium']}</div></div>""", unsafe_allow_html=True)
                g2.markdown(f"""<div class="card"><div class="card-title">Breakeven Spot</div><div class="card-value" style="font-size:20px;">INR {data['breakeven']}</div></div>""", unsafe_allow_html=True)
                g3.markdown(f"""<div class="card"><div class="card-title">Implied Vol (IV)</div><div class="card-value" style="font-size:20px;">{pi.get('iv', 'N/A')}%</div></div>""", unsafe_allow_html=True)
                g4.markdown(f"""<div class="card"><div class="card-title">Est. Delta</div><div class="card-value" style="font-size:20px;">{pi.get('delta', 'N/A')}</div></div>""", unsafe_allow_html=True)
                g5.markdown(f"""<div class="card"><div class="card-title">Est. Theta / Day</div><div class="card-value" style="font-size:20px; color:#FF5252;">INR {pi.get('theta_per_day', 'N/A')}</div></div>""", unsafe_allow_html=True)

                premium_sl_val = round(pi['premium'] * (premium_sl_pct / 100), 2)
                max_loss_per_lot = premium_sl_val * lot_size
                risk_amount = capital * (risk_pct / 100)
                suggested_lots = int(risk_amount // max_loss_per_lot) if max_loss_per_lot > 0 else 0

                s1, s2, s3 = st.columns(3)
                s1.markdown(f"""<div class="card"><div class="card-title">Risk Capital Budget</div><div class="card-value">INR {risk_amount:,.0f}</div></div>""", unsafe_allow_html=True)
                s2.markdown(f"""<div class="card"><div class="card-title">Max Loss Per Lot</div><div class="card-value" style="color:#FF5252;">INR {max_loss_per_lot:,.0f}</div></div>""", unsafe_allow_html=True)
                s3.markdown(f"""<div class="card"><div class="card-title">Recommended Lots</div><div class="card-value" style="color:#00E676;">{suggested_lots} Lots</div></div>""", unsafe_allow_html=True)

        st.markdown("---")

        # --- CHART & TECHNICAL METRICS HUB ---
        chart_col, tech_col = st.columns([2.5, 1])

        with chart_col:
            st.markdown("### Price Action & Technical Overlay")
            fig = render_candlestick_chart(data["df_5m"], data["symbol"], data["orb_high"], data["orb_low"])
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        with tech_col:
            st.markdown("### Live Indicators")
            st.markdown(f"""
            <div class="card">
                <div class="card-title">Session VWAP</div>
                <div class="card-value">INR {data['vwap']:,.2f}</div>
            </div>
            <div class="card">
                <div class="card-title">Supertrend (7, 3.0)</div>
                <div class="card-value">INR {data['supertrend']:,.2f}</div>
            </div>
            <div class="card">
                <div class="card-title">RSI (14)</div>
                <div class="card-value">{data['rsi']:.1f}</div>
            </div>
            <div class="card">
                <div class="card-title">ATR Volatility (14)</div>
                <div class="card-value">INR {data['atr']:,.2f}</div>
            </div>
            <div class="card">
                <div class="card-title">SMC Imbalance (FVG)</div>
                <div class="card-value" style="font-size:16px;">{data['fvg_desc']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # --- TIMEFRAMES & CHECKLIST ---
        col_mtf, col_check = st.columns(2)

        with col_mtf:
            st.markdown("### Multi-Timeframe Alignment")
            mtf_df = pd.DataFrame([data["trends"]]).T
            mtf_df.columns = ["Trend Direction"]
            st.dataframe(mtf_df, use_container_width=True)

        with col_check:
            st.markdown("### Quantitative System Checklist")
            for check, (passed, desc) in data["checks"].items():
                badge_style = "check-pass" if passed else "check-fail"
                status_tag = "[PASS]" if passed else "[FAIL]"
                st.markdown(f"""
                <div class="{badge_style}">
                    <b>{status_tag} {check}:</b> {desc}
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # --- TELEGRAM & LOGGING ---
        act1, act2 = st.columns(2)
        
        with act1:
            st.markdown("### Telegram Dispatcher")
            if data['bias'] != "NEUTRAL / NO TRADE" and data.get("premium_info") and data["premium_info"].get("premium"):
                prem_entry = float(data["premium_info"]["premium"])
                prem_sl = round(prem_entry * (1 - (premium_sl_pct / 100)), 2)
                prem_t1 = round(prem_entry * 1.30, 2)
                prem_t2 = round(prem_entry * 1.50, 2)
                
                src_type = data['premium_info'].get('source', '')
                src_tag = f" ({src_type})" if src_type else ""
                prem_entry_str = f"INR {prem_entry:,.2f}{src_tag}"
                prem_sl_str = f"INR {prem_sl:,.2f} (-{premium_sl_pct}%)"
                prem_target_str = f"INR {prem_t1:,.2f} / INR {prem_t2:,.2f} (+30% / +50%)"
            else:
                prem_entry_str, prem_sl_str, prem_target_str = "N/A", "N/A", "N/A"

            alert_msg = (
                f"AI TRADING SIGNAL: {data['symbol']}\n\n"
                f"- Decision: {data['bias']}\n"
                f"- Recommended Strike: {data['strike']}\n\n"
                f"OPTION PREMIUM LEVELS:\n"
                f"- Entry Premium: {prem_entry_str}\n"
                f"- Stop Loss: {prem_sl_str}\n"
                f"- Targets: {prem_target_str}\n\n"
                f"SPOT INDEX LEVELS:\n"
                f"- Spot Entry: INR {data['price']:,.2f}\n"
                f"- Spot SL: INR {data['sl']:,.2f}\n"
                f"- Spot Targets: INR {data['target1']:,.2f} / INR {data['target2']:,.2f}\n\n"
                f"SIGNAL METRICS:\n"
                f"- Risk-Reward: {data['rrr']}\n"
                f"- Confidence Score: {data['score']}%\n"
                f"- PCR: {data['pcr'] if data['pcr'] is not None else 'N/A'}\n\n"
                f"Timestamp: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}"
            )
            st.text_area("Telegram Message Preview", value=alert_msg, height=220)
            if st.button("Send Alert to Telegram Channel", use_container_width=True):
                if send_telegram_alert(alert_msg):
                    st.success("Signal broadcasted to Telegram successfully!")

        with act2:
            st.markdown("### Journal Entry Logger")
            notes = st.text_input("Trade Notes", value=f"{data['bias']} setup logged for {data['symbol']}")
            if st.button("Save Trade to Database", use_container_width=True):
                log_trade(
                    symbol=data['symbol'], price=data['price'], bias=data['bias'], strike=data['strike'],
                    entry=data['price'], sl=data['sl'], target=data['target1'], score=data['score'],
                    pcr=data['pcr'] if data['pcr'] is not None else -1, vix=vix_val if vix_val is not None else -1,
                    notes=notes
                )
                st.success(f"Trade successfully recorded in {DB_NAME} database!")

    else:
        st.warning(f"Unable to fetch data for '{user_symbol}'. Try symbols like NIFTY, BANKNIFTY, SENSEX, RELIANCE, TCS.")

with tab_journal:
    st.markdown("### Trade History Database")
    journal_df = get_journal_data()
    if not journal_df.empty:
        st.dataframe(journal_df, use_container_width=True)
        csv = journal_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Journal as CSV", data=csv,
                            file_name=f"trade_journal_{datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv")
    else:
        st.info("No trades logged in database yet.")
