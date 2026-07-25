import math
import time
import sqlite3
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from streamlit_autorefresh import st_autorefresh

IST = ZoneInfo("Asia/Kolkata")

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Trading Assistant V4",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS STYLING ---
st.markdown("""
<style>
    .buy-call { color: #00E676; font-size: 28px; font-weight: bold; }
    .buy-put { color: #FF5252; font-size: 28px; font-weight: bold; }
    .neutral { color: #FFD600; font-size: 28px; font-weight: bold; }
    .warn-box {
        background-color: #3A2A00; border: 1px solid #FFD600;
        padding: 10px 14px; border-radius: 8px; margin-bottom: 10px;
    }
    .danger-box {
        background-color: #3A0000; border: 1px solid #FF5252;
        padding: 10px 14px; border-radius: 8px; margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================================
# LOT SIZES — NSE revises these periodically (last major revision Jan 2026).
# ALWAYS verify current lot sizes at nseindia.com before sizing a real trade.
# These are pre-fills only; the sidebar lets you override them.
# =========================================================================
DEFAULT_LOT_SIZES = {
    "NIFTY": 65,
    "NIFTY50": 65,
    "BANKNIFTY": 30,
    "FINNIFTY": 60,
    "MIDCPNIFTY": 140,
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
        st.error("⚠️ Telegram credentials missing in Streamlit Secrets!")
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
# 3. MARKET HOURS / SESSION HELPERS
# =========================================================================
def market_status():
    """Returns (is_open, label). NSE cash/F&O: 9:15-15:30 IST, Mon-Fri.
    NOTE: this does NOT account for NSE trading holidays — cross-check
    the NSE holiday calendar for full accuracy."""
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False, "Market Closed (Weekend)"
    open_t = now.replace(hour=9, minute=15, second=0, microsecond=0)
    close_t = now.replace(hour=15, minute=30, second=0, microsecond=0)
    if now < open_t:
        return False, "Market Closed (Pre-Open) — showing last session data"
    if now > close_t:
        return False, "Market Closed — showing last session data"
    return True, "Market Open"

# =========================================================================
# 4. TECHNICAL CALCULATIONS
# =========================================================================
def calculate_session_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP anchored to the CURRENT trading session only (resets daily),
    not cumulative across the whole lookback window."""
    idx = df.index
    if idx.tz is not None:
        session_date = idx.tz_convert(IST).date
    else:
        session_date = idx.date
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

def opening_range(df_today: pd.DataFrame, minutes: int = 15):
    """First N minutes of the session high/low (Opening Range Breakout ref)."""
    if df_today.empty:
        return None, None
    idx = df_today.index
    session_date = idx.tz_convert(IST).date()[0] if idx.tz is not None else idx.date[0]
    day_open = datetime.combine(session_date, datetime.min.time(), tzinfo=IST).replace(hour=9, minute=15)
    day_open_end = day_open + timedelta(minutes=minutes)
    if idx.tz is not None:
        mask = (idx.tz_convert(IST) >= day_open) & (idx.tz_convert(IST) < day_open_end)
    else:
        mask = (idx >= day_open.replace(tzinfo=None)) & (idx < day_open_end.replace(tzinfo=None))
    orb_slice = df_today[mask]
    if orb_slice.empty:
        return None, None
    return orb_slice['High'].max(), orb_slice['Low'].min()

# =========================================================================
# 5. INDIA VIX — VOLATILITY REGIME FILTER
# =========================================================================
@st.cache_data(ttl=60)
def get_india_vix():
    try:
        vix_df = yf.Ticker("^INDIAVIX").history(period="5d", interval="15m")
        if vix_df.empty:
            return None, "Unavailable"
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
        return None, "Unavailable"

# =========================================================================
# 6. REAL NSE OPTION CHAIN (OI, PCR, Max Pain, IV, ATM premium)
# NOTE: nseindia.com aggressively rate-limits / blocks datacenter & cloud
# IPs (including Streamlit Community Cloud) with Cloudflare-style checks.
# This WILL sometimes fail when deployed — the app falls back to
# "Option data unavailable" rather than faking a neutral PCR, so you can
# always tell whether a signal used real options data or not.
# For reliable production use, replace this with a paid vendor
# (Kite Connect / Upstox / Sensibull / TrueData / Global Datafeeds API).
# =========================================================================
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
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

        if is_index:
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
        else:
            url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"

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
            ce = r.get("CE")
            pe = r.get("PE")
            strike = r.get("strikePrice")
            if ce:
                total_call_oi += ce.get("openInterest", 0)
            if pe:
                total_put_oi += pe.get("openInterest", 0)
            strikes.append({"strike": strike, "CE": ce, "PE": pe})

        pcr = round(total_put_oi / total_call_oi, 2) if total_call_oi > 0 else None

        # ATM strike = strike closest to underlying value
        atm_row = min(strikes, key=lambda x: abs(x["strike"] - underlying))

        # Max Pain: strike where total option-writer payout is minimized
        max_pain_strike, min_loss = None, None
        for s in strikes:
            loss = 0
            for s2 in strikes:
                if s2["CE"]:
                    itm = max(0, s["strike"] - s2["strike"])
                    loss += itm * s2["CE"].get("openInterest", 0)
                if s2["PE"]:
                    itm = max(0, s2["strike"] - s["strike"])
                    loss += itm * s2["PE"].get("openInterest", 0)
            if min_loss is None or loss < min_loss:
                min_loss = loss
                max_pain_strike = s["strike"]

        return {
            "underlying": underlying,
            "expiry": nearest_expiry,
            "pcr": pcr,
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
            "atm_strike": atm_row["strike"],
            "atm_ce": atm_row["CE"],
            "atm_pe": atm_row["PE"],
            "max_pain": max_pain_strike,
        }
    except Exception:
        return None

def is_expiry_today(expiry_str: str) -> bool:
    try:
        expiry_date = datetime.strptime(expiry_str, "%d-%b-%Y").date()
        return expiry_date == datetime.now(IST).date()
    except Exception:
        return False

# =========================================================================
# 7. BLACK-SCHOLES FALLBACK (used only if live NSE premium/IV unavailable)
# =========================================================================
def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def _norm_pdf(x):
    return math.exp(-x ** 2 / 2) / math.sqrt(2 * math.pi)

def bs_price_greeks(S, K, T_years, sigma, option_type, r=0.07):
    if T_years <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T_years) / (sigma * math.sqrt(T_years))
    d2 = d1 - sigma * math.sqrt(T_years)
    if option_type == "CE":
        price = S * _norm_cdf(d1) - K * math.exp(-r * T_years) * _norm_cdf(d2)
        delta = _norm_cdf(d1)
    else:
        price = K * math.exp(-r * T_years) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
        delta = _norm_cdf(d1) - 1
    theta_per_day = (-(S * _norm_pdf(d1) * sigma) / (2 * math.sqrt(T_years)) -
                      (r * K * math.exp(-r * T_years) *
                       (_norm_cdf(d2) if option_type == "CE" else _norm_cdf(-d2)))) / 365
    return {"price": round(price, 2), "delta": round(delta, 3), "theta_per_day": round(theta_per_day, 2)}

# =========================================================================
# 8. CORE ANALYSIS ENGINE
# =========================================================================
def get_atm_display_strike(symbol: str, spot_price: float, bias: str, nse_chain) -> str:
    if nse_chain and nse_chain.get("atm_strike"):
        strike = nse_chain["atm_strike"]
    else:
        if "BANKNIFTY" in symbol:
            step = 100
        elif "NIFTY" in symbol:
            step = 50
        elif "SENSEX" in symbol:
            step = 100
        else:
            step = 10
        strike = round(spot_price / step) * step
    option_type = "CE" if bias == "BUY CALL" else "PE"
    return f"{int(strike)} {option_type}"

@st.cache_data(ttl=20)
def analyze_market(symbol: str, vix_val, orb_confirm_pct: float):
    raw_sym = symbol.upper().strip()
    is_index = raw_sym in NSE_INDEX_SYMBOLS

    index_map = {"NIFTY": "^NSEI", "NIFTY50": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN"}
    ticker_sym = index_map.get(raw_sym, raw_sym)
    if not ticker_sym.startswith("^") and not ticker_sym.endswith(".NS"):
        ticker_sym += ".NS"

    try:
        ticker = yf.Ticker(ticker_sym)
        df_5m = ticker.history(period="5d", interval="5m")
        df_15m = ticker.history(period="10d", interval="15m")
        df_1h = ticker.history(period="1mo", interval="1h")
        df_1d = ticker.history(period="6mo", interval="1d")

        if df_5m.empty or len(df_5m) < 20:
            return None

        current_price = df_5m['Close'].iloc[-1]
        prev_close = df_1d['Close'].iloc[-2] if len(df_1d) > 1 else current_price
        day_high = df_5m['High'].max()
        day_low = df_5m['Low'].min()
        price_change_pct = ((current_price - prev_close) / prev_close) * 100

        trend_1d = "BULLISH" if df_1d['Close'].iloc[-1] > df_1d['Close'].rolling(20).mean().iloc[-1] else "BEARISH"
        trend_1h = "BULLISH" if df_1h['Close'].iloc[-1] > df_1h['Close'].rolling(20).mean().iloc[-1] else "BEARISH"
        trend_15m = "BULLISH" if df_15m['Close'].iloc[-1] > df_15m['Close'].rolling(20).mean().iloc[-1] else "BEARISH"
        trend_5m = "BULLISH" if df_5m['Close'].iloc[-1] > df_5m['Close'].rolling(20).mean().iloc[-1] else "BEARISH"

        # --- Session-anchored VWAP (fixed) ---
        df_5m['VWAP'] = calculate_session_vwap(df_5m)
        current_vwap = df_5m['VWAP'].iloc[-1]
        above_vwap = current_price > current_vwap

        # --- ATR for volatility-adjusted risk ---
        df_5m['ATR'] = calculate_atr(df_5m, 14)
        current_atr = df_5m['ATR'].iloc[-1]
        atr_avg = df_5m['ATR'].rolling(20).mean().iloc[-1]
        atr_expanding = bool(current_atr and atr_avg and current_atr > atr_avg)
        if not current_atr or np.isnan(current_atr):
            current_atr = current_price * 0.003  # safety fallback ~0.3%

        # --- Opening Range Breakout ---
        idx = df_5m.index
        today_key = idx.tz_convert(IST).date() if idx.tz is not None else idx.date
        last_day = today_key[-1]
        df_today = df_5m[today_key == last_day] if hasattr(today_key, "__eq__") else df_5m
        try:
            df_today = df_5m[[d == last_day for d in today_key]]
        except Exception:
            df_today = df_5m.tail(75)
        orb_high, orb_low = opening_range(df_today, minutes=15)
        orb_bull = orb_high is not None and current_price > orb_high
        orb_bear = orb_low is not None and current_price < orb_low

        # --- Real NSE option chain (graceful fallback) ---
        nse_chain = fetch_nse_option_chain(raw_sym if is_index else raw_sym, is_index)
        pcr = nse_chain["pcr"] if nse_chain and nse_chain["pcr"] is not None else None
        expiry_today = is_expiry_today(nse_chain["expiry"]) if nse_chain else False

        # ==================== CONFIDENCE SCORE (max 100) ====================
        score = 0
        checks = {}

        aligned_bullish = (trend_1d == "BULLISH" and trend_1h == "BULLISH" and trend_15m == "BULLISH")
        aligned_bearish = (trend_1d == "BEARISH" and trend_1h == "BEARISH" and trend_15m == "BEARISH")

        if aligned_bullish or aligned_bearish:
            score += 25
            checks["MTF Alignment"] = (True, "1D, 1H, 15M Aligned")
        else:
            checks["MTF Alignment"] = (False, "Timeframe Conflict")

        if (aligned_bullish and above_vwap) or (aligned_bearish and not above_vwap):
            score += 20
            checks["Session VWAP"] = (True, f"{'Above' if above_vwap else 'Below'} session VWAP")
        else:
            checks["Session VWAP"] = (False, "Conflicting Price vs Session VWAP")

        if (aligned_bullish and orb_bull) or (aligned_bearish and orb_bear):
            score += 15
            checks["Opening Range Breakout"] = (True, "Price beyond opening range in trend direction")
        else:
            checks["Opening Range Breakout"] = (False, "No confirmed ORB breakout")

        if atr_expanding:
            score += 15
            checks["Volatility Expansion (ATR)"] = (True, "ATR rising — momentum active")
        else:
            checks["Volatility Expansion (ATR)"] = (False, "ATR flat/contracting")

        if pcr is None:
            checks["Option Chain (PCR)"] = (False, "NSE option data unavailable — not scored")
        elif (aligned_bullish and pcr >= 1.0) or (aligned_bearish and pcr < 1.0):
            score += 25
            checks["Option Chain (PCR)"] = (True, f"PCR = {pcr}")
        else:
            checks["Option Chain (PCR)"] = (False, f"PCR = {pcr} (Divergence)")

        # --- Volatility regime gate (VIX) ---
        vix_block = vix_val is not None and vix_val >= 22

        # --- Expiry-day time cutoff for fresh option buying ---
        now_ist = datetime.now(IST)
        expiry_cutoff = expiry_today and now_ist.hour >= 14 and now_ist.minute >= 30

        bias, strike, sl, target1, target2 = "NEUTRAL / NO TRADE", "N/A", 0.0, 0.0, 0.0
        premium_info = None

        if score >= 70 and aligned_bullish and above_vwap and not vix_block and not expiry_cutoff:
            bias = "BUY CALL"
        elif score >= 70 and aligned_bearish and not above_vwap and not vix_block and not expiry_cutoff:
            bias = "BUY PUT"

        if bias != "NEUTRAL / NO TRADE":
            strike = get_atm_display_strike(raw_sym, current_price, bias, nse_chain)
            # ATR-based spot SL/targets instead of fixed %
            atr_mult_sl, atr_mult_t1, atr_mult_t2 = 1.0, 1.5, 2.5
            if bias == "BUY CALL":
                sl = round(current_price - atr_mult_sl * current_atr, 2)
                target1 = round(current_price + atr_mult_t1 * current_atr, 2)
                target2 = round(current_price + atr_mult_t2 * current_atr, 2)
            else:
                sl = round(current_price + atr_mult_sl * current_atr, 2)
                target1 = round(current_price - atr_mult_t1 * current_atr, 2)
                target2 = round(current_price - atr_mult_t2 * current_atr, 2)

            # Premium & Greeks: prefer REAL NSE premium/IV, else Black-Scholes fallback
            leg = nse_chain["atm_ce"] if (nse_chain and bias == "BUY CALL") else (nse_chain["atm_pe"] if nse_chain else None)
            if leg and leg.get("lastPrice"):
                iv = leg.get("impliedVolatility", 0) / 100
                premium_info = {
                    "source": "NSE live",
                    "premium": leg.get("lastPrice"),
                    "iv": leg.get("impliedVolatility"),
                    "oi": leg.get("openInterest"),
                    "chg_oi": leg.get("changeinOpenInterest"),
                }
            elif nse_chain:
                try:
                    expiry_date = datetime.strptime(nse_chain["expiry"], "%d-%b-%Y").date()
                    days_to_expiry = max((expiry_date - now_ist.date()).days, 0) + 0.25
                    T = days_to_expiry / 365
                    est_iv = 0.14 if is_index else 0.30
                    bs = bs_price_greeks(current_price, nse_chain["atm_strike"], T, est_iv,
                                          "CE" if bias == "BUY CALL" else "PE")
                    if bs:
                        premium_info = {"source": "Black-Scholes estimate (est. IV)", "premium": bs["price"],
                                         "iv": est_iv * 100, "delta": bs["delta"], "theta_per_day": bs["theta_per_day"]}
                except Exception:
                    premium_info = None

        return {
            "symbol": raw_sym, "price": current_price, "change_pct": price_change_pct,
            "high": day_high, "low": day_low, "vwap": current_vwap, "atr": current_atr,
            "pcr": pcr, "bias": bias, "strike": strike, "sl": sl, "target1": target1, "target2": target2,
            "score": score, "checks": checks,
            "trends": {"1D": trend_1d, "1H": trend_1h, "15M": trend_15m, "5M": trend_5m},
            "orb_high": orb_high, "orb_low": orb_low,
            "nse_chain": nse_chain, "expiry_today": expiry_today, "expiry_cutoff": expiry_cutoff,
            "vix_block": vix_block, "premium_info": premium_info,
        }
    except Exception:
        return None

# =========================================================================
# STREAMLIT DASHBOARD UI
# =========================================================================
st.title("⚡ AI Trading Assistant V4")
st.markdown("Session-VWAP • Real NSE Option Chain • ATR Risk • India VIX Regime • ORB • Expiry-Aware")

is_open, status_label = market_status()
if not is_open:
    st.markdown(f'<div class="warn-box">🕒 <b>{status_label}</b></div>', unsafe_allow_html=True)

vix_val, vix_regime = get_india_vix()
vix_col1, vix_col2 = st.columns([1, 3])
with vix_col1:
    st.metric("India VIX", vix_val if vix_val is not None else "N/A")
with vix_col2:
    st.markdown(f"**Volatility Regime:** {vix_regime}")

st.sidebar.header("⚙️ Configuration")
user_symbol = st.sidebar.text_input("Symbol (e.g. NIFTY, SENSEX, RELIANCE)", value="NIFTY").upper().strip()

st.sidebar.subheader("🔄 Continuous Market Auto-Refresh")
enable_autorefresh = st.sidebar.checkbox("Enable Auto-Refresh", value=False)
refresh_interval = st.sidebar.slider("Refresh Timer (Seconds)", min_value=15, max_value=120, value=30)
if enable_autorefresh:
    st_autorefresh(interval=refresh_interval * 1000, key="market_auto_refresh")

st.sidebar.divider()
st.sidebar.subheader("💰 Position Sizing (Option Buying)")
st.sidebar.caption("Lot sizes are revised periodically by NSE — verify at nseindia.com before trading.")
capital = st.sidebar.number_input("Trading Capital (₹)", min_value=1000, value=100000, step=1000)
risk_pct = st.sidebar.slider("Max Risk per Trade (%)", 0.5, 5.0, 1.5, 0.5)
default_lot = DEFAULT_LOT_SIZES.get(user_symbol, 1)
lot_size = st.sidebar.number_input("Lot Size (verify current value)", min_value=1, value=default_lot, step=1)
premium_sl_pct = st.sidebar.slider("Premium Stop-Loss (%)", 10, 50, 30, 5,
                                    help="Common practice: exit an option buy if premium falls this % from entry.")

st.sidebar.divider()
st.sidebar.caption(
    "⚠️ Educational tool only. Not SEBI-registered investment advice. "
    "Signals are rule-based, not guaranteed. Verify option data, lot sizes "
    "and NSE holidays independently before trading real capital."
)

tab_screener, tab_journal = st.tabs(["📊 Live Market Signals", "📓 Trade Journal"])

with tab_screener:
    data = analyze_market(user_symbol, vix_val, orb_confirm_pct=0.0)

    if data:
        if data["expiry_today"]:
            st.markdown('<div class="warn-box">📅 <b>Today is Expiry Day</b> — gamma/theta risk is elevated. '
                        'Fresh option buying is auto-blocked after 2:30 PM IST.</div>', unsafe_allow_html=True)
        if data["vix_block"]:
            st.markdown('<div class="danger-box">🌪️ <b>India VIX is HIGH</b> — fresh naked option buying is '
                        'suppressed. Consider spreads or standing aside.</div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns([1.5, 1.2, 1.2, 1.5])
        with col1:
            st.markdown("**Signal Bias**")
            if data["bias"] == "BUY CALL":
                st.markdown(f'<div class="buy-call">🟢 {data["bias"]}</div>', unsafe_allow_html=True)
            elif data["bias"] == "BUY PUT":
                st.markdown(f'<div class="buy-put">🔴 {data["bias"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="neutral">🟡 {data["bias"]}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown("**Confidence Score**")
            st.progress(data["score"] / 100)
            st.markdown(f"### **{data['score']}%**")
        with col3:
            st.markdown("**Put-Call Ratio (PCR)**")
            if data["pcr"] is None:
                st.markdown("### ⚪ N/A")
                st.caption("NSE option data unavailable this refresh")
            else:
                pcr_icon = "🟢" if data["pcr"] >= 1.0 else "🔴"
                st.markdown(f"### {pcr_icon} **{data['pcr']}**")
        with col4:
            st.markdown("**Spot Price**")
            st.metric(label=f"{data['symbol']}", value=f"₹{data['price']:,.2f}", delta=f"{data['change_pct']:+.2f}%")

        st.divider()

        if data["bias"] != "NEUTRAL / NO TRADE":
            st.subheader("🎯 Trade Execution Parameters (Spot-Level, ATR-based)")
            p1, p2, p3, p4 = st.columns(4)
            p1.markdown(f"**Recommended Strike**\n### `{data['strike']}`")
            p2.markdown(f"**Entry Spot Price**\n### ₹{data['price']:,.2f}")
            p3.markdown(f"**Stop Loss (SL)**\n### ₹{data['sl']:,.2f}")
            p4.markdown(f"**Target 1 / Target 2**\n### ₹{data['target1']:,.2f} / ₹{data['target2']:,.2f}")
            st.caption(f"SL/targets sized off ATR(14) = ₹{data['atr']:.2f}, not a fixed %.")

            if data["premium_info"]:
                pi = data["premium_info"]
                st.subheader("💊 Option Premium & Greeks")
                g1, g2, g3, g4 = st.columns(4)
                g1.markdown(f"**Entry Premium**\n### ₹{pi['premium']}")
                g2.markdown(f"**Implied Vol**\n### {pi.get('iv', 'N/A')}%")
                if "delta" in pi:
                    g3.markdown(f"**Est. Delta**\n### {pi['delta']}")
                    g4.markdown(f"**Est. Theta/day**\n### ₹{pi['theta_per_day']}")
                else:
                    g3.markdown(f"**OI**\n### {pi.get('oi', 'N/A')}")
                    g4.markdown(f"**Chg in OI**\n### {pi.get('chg_oi', 'N/A')}")
                st.caption(f"Source: {pi['source']}")

                premium_sl_val = round(pi['premium'] * (premium_sl_pct / 100), 2)
                max_loss_per_lot = premium_sl_val * lot_size
                risk_amount = capital * (risk_pct / 100)
                suggested_lots = int(risk_amount // max_loss_per_lot) if max_loss_per_lot > 0 else 0

                st.subheader("📐 Suggested Position Size")
                s1, s2, s3 = st.columns(3)
                s1.markdown(f"**Risk Budget**\n### ₹{risk_amount:,.0f}")
                s2.markdown(f"**Max Loss / Lot**\n### ₹{max_loss_per_lot:,.0f}")
                s3.markdown(f"**Suggested Lots**\n### {suggested_lots}")
                if suggested_lots == 0:
                    st.warning("Risk budget is smaller than 1 lot's worst-case premium loss at this SL% — reduce lot size or widen capital.")

            if data["nse_chain"] and data["nse_chain"].get("max_pain"):
                st.caption(f"Max Pain (nearest expiry {data['nse_chain']['expiry']}): "
                           f"₹{data['nse_chain']['max_pain']}")
            st.divider()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Session VWAP", f"₹{data['vwap']:,.2f}")
        m2.metric("Day High", f"₹{data['high']:,.2f}")
        m3.metric("Day Low", f"₹{data['low']:,.2f}")
        m4.metric("ATR (14, 5m)", f"₹{data['atr']:,.2f}")

        if data["orb_high"] and data["orb_low"]:
            o1, o2 = st.columns(2)
            o1.metric("Opening Range High (first 15m)", f"₹{data['orb_high']:,.2f}")
            o2.metric("Opening Range Low (first 15m)", f"₹{data['orb_low']:,.2f}")

        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📊 Multi-Timeframe Alignment")
            t_df = pd.DataFrame([data["trends"]]).T
            t_df.columns = ["Direction"]
            st.table(t_df)
        with c2:
            st.subheader("🎯 System Checklist")
            for check, (passed, desc) in data["checks"].items():
                icon = "🟢" if passed else "🔴"
                st.write(f"{icon} **{check}**: {desc}")

        st.divider()

        act1, act2 = st.columns(2)
        with act1:
            st.subheader("📲 Telegram Dispatcher")
            premium_line = ""
            if data["premium_info"]:
                premium_line = f"• *Est. Premium:* ₹{data['premium_info']['premium']} ({data['premium_info']['source']})\n"
            alert_msg = (
                f"🚨 *AI TRADING SIGNAL: {data['symbol']}*\n\n"
                f"• *Decision:* `{data['bias']}`\n"
                f"• *Recommended Strike:* `{data['strike']}`\n"
                f"• *Entry Price:* ₹{data['price']:,.2f}\n"
                f"• *Stop Loss:* ₹{data['sl']:,.2f}\n"
                f"• *Target 1:* ₹{data['target1']:,.2f}\n"
                f"{premium_line}"
                f"• *Confidence Score:* `{data['score']}%`\n"
                f"• *PCR:* `{data['pcr'] if data['pcr'] is not None else 'N/A'}`\n"
                f"• *India VIX:* `{vix_val if vix_val is not None else 'N/A'}` ({vix_regime})\n\n"
                f"⏰ _Timestamp: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}_\n"
                f"⚠️ Educational signal, not investment advice."
            )
            st.text_area("Telegram Preview", value=alert_msg, height=160)
            if st.button("🚀 Send Alert to Telegram"):
                if send_telegram_alert(alert_msg):
                    st.success("✅ Signal broadcasted to Telegram successfully!")

        with act2:
            st.subheader("📓 Journal Entry Logging")
            st.caption("⚠️ SQLite on Streamlit Community Cloud is not persistent across "
                       "redeploys/restarts. Export CSV regularly from the Journal tab as backup.")
            notes = st.text_input("Custom Notes", value=f"{data['bias']} setup logged for {data['symbol']}")
            if st.button("💾 Log Trade to Database"):
                log_trade(
                    symbol=data['symbol'], price=data['price'], bias=data['bias'], strike=data['strike'],
                    entry=data['price'], sl=data['sl'], target=data['target1'], score=data['score'],
                    pcr=data['pcr'] if data['pcr'] is not None else -1, vix=vix_val if vix_val is not None else -1,
                    notes=notes
                )
                st.success(f"✅ Trade recorded in `{DB_NAME}`!")
    else:
        st.warning(f"Unable to fetch data for '{user_symbol}'. Try NIFTY, BANKNIFTY, SENSEX, or a stock ticker like RELIANCE, TCS, INFY.")

with tab_journal:
    st.subheader("📓 Trade Journal History")
    journal_df = get_journal_data()
    if not journal_df.empty:
        st.dataframe(journal_df, use_container_width=True)
        st.caption(f"Total Entries Logged: {len(journal_df)}")
        csv = journal_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export Journal as CSV (backup)", data=csv,
                            file_name=f"trade_journal_{datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv")
    else:
        st.info("No trade entries logged yet. Log your first setup from the Live Market Signals tab!")
