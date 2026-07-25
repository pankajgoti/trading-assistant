import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import sqlite3
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Trading Assistant V3",
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
    .strike-box {
        background-color: #1E222D;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #2A2E39;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        color: #00E676;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. SQLITE TRADE JOURNAL DATABASE ---
DB_NAME = "trade_journal.db"

def init_db():
    """Initializes SQLite database for trade logging."""
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
            notes TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_trade(symbol, price, bias, strike, entry, sl, target, score, pcr, notes="Auto-Logged"):
    """Saves a trade setup to SQLite database."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        INSERT INTO journal (timestamp, symbol, price, bias, strike, entry, sl, target, score, pcr, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), symbol, price, bias, strike, entry, sl, target, score, pcr, notes))
    conn.commit()
    conn.close()

def get_journal_data():
    """Fetches logged trades from SQLite."""
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM journal ORDER BY id DESC", conn)
    conn.close()
    return df

init_db()

# --- 2. TELEGRAM ALERT DISPATCHER ---
def send_telegram_alert(message: str) -> bool:
    """Delivers formatted trading signals to Telegram."""
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

# --- 3. TECHNICAL CALCULATIONS & OPTION STRIKE ENGINE ---
def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculates intraday Volume Weighted Average Price (VWAP)."""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    return (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()

def get_option_chain_pcr(ticker_obj) -> float:
    """Calculates Put-Call Ratio (PCR) from Open Interest."""
    try:
        expirations = ticker_obj.options
        if not expirations:
            return 1.0
        
        chain = ticker_obj.option_chain(expirations[0])
        total_call_oi = chain.calls['openInterest'].sum()
        total_put_oi = chain.puts['openInterest'].sum()
        
        if total_call_oi > 0:
            return round(total_put_oi / total_call_oi, 2)
        return 1.0
    except Exception:
        return 1.0

def get_atm_strike(symbol: str, spot_price: float, bias: str) -> str:
    """Calculates At-The-Money (ATM) option strike for Stocks & Indices."""
    if "NIFTY" in symbol and "BANK" not in symbol:
        step = 50
    elif "BANKNIFTY" in symbol or "SENSEX" in symbol:
        step = 100
    else:
        step = 10  # Standard step for equities
        
    atm_strike = round(spot_price / step) * step
    option_type = "CE" if bias == "BUY CALL" else "PE"
    return f"{int(atm_strike)} {option_type}"

@st.cache_data(ttl=20)
def analyze_market(symbol: str):
    """Executes multi-timeframe trend, VWAP, PCR, and signal generation."""
    raw_sym = symbol.upper().strip()
    
    # Symbol mapping for indices
    index_map = {
        "NIFTY": "^NSEI",
        "NIFTY50": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "SENSEX": "^BSESN"
    }
    
    ticker_sym = index_map.get(raw_sym, raw_sym)
    if not ticker_sym.startswith("^") and not ticker_sym.endswith(".NS"):
        ticker_sym += ".NS"

    try:
        ticker = yf.Ticker(ticker_sym)
        
        # Fetch multi-timeframe candles
        df_5m = ticker.history(period="5d", interval="5m")
        df_15m = ticker.history(period="10d", interval="15m")
        df_1h = ticker.history(period="1mo", interval="1h")
        df_1d = ticker.history(period="6mo", interval="1d")

        if df_5m.empty or len(df_5m) < 20:
            return None

        # Price metrics
        current_price = df_5m['Close'].iloc[-1]
        prev_close = df_1d['Close'].iloc[-2] if len(df_1d) > 1 else current_price
        day_high = df_5m['High'].max()
        day_low = df_5m['Low'].min()
        price_change_pct = ((current_price - prev_close) / prev_close) * 100

        # Trends across timeframes
        trend_1d = "BULLISH" if df_1d['Close'].iloc[-1] > df_1d['Close'].rolling(20).mean().iloc[-1] else "BEARISH"
        trend_1h = "BULLISH" if df_1h['Close'].iloc[-1] > df_1h['Close'].rolling(20).mean().iloc[-1] else "BEARISH"
        trend_15m = "BULLISH" if df_15m['Close'].iloc[-1] > df_15m['Close'].rolling(20).mean().iloc[-1] else "BEARISH"
        trend_5m = "BULLISH" if df_5m['Close'].iloc[-1] > df_5m['Close'].rolling(20).mean().iloc[-1] else "BEARISH"

        # VWAP & Volume Surge
        df_5m['VWAP'] = calculate_vwap(df_5m)
        current_vwap = df_5m['VWAP'].iloc[-1]
        above_vwap = current_price > current_vwap

        recent_vol = df_5m['Volume'].iloc[-1]
        avg_vol = df_5m['Volume'].rolling(20).mean().iloc[-1]
        high_vol = recent_vol > (1.5 * avg_vol)

        # Put-Call Ratio
        pcr = get_option_chain_pcr(ticker)

        # Confidence Score Logic
        score = 0
        checks = {}

        aligned_bullish = (trend_1d == "BULLISH" and trend_1h == "BULLISH" and trend_15m == "BULLISH")
        aligned_bearish = (trend_1d == "BEARISH" and trend_1h == "BEARISH" and trend_15m == "BEARISH")
        
        if aligned_bullish or aligned_bearish:
            score += 30
            checks["MTF Alignment"] = (True, "1D, 1H, 15M Aligned")
        else:
            checks["MTF Alignment"] = (False, "Timeframe Conflict")

        if (aligned_bullish and above_vwap) or (aligned_bearish and not above_vwap):
            score += 25
            checks["VWAP Confirmation"] = (True, f"{'Above' if above_vwap else 'Below'} VWAP")
        else:
            checks["VWAP Confirmation"] = (False, "Conflicting Price vs VWAP")

        if high_vol or ticker_sym.startswith("^"):
            score += 20
            checks["Volume / Momentum"] = (True, "Momentum Active")
        else:
            checks["Volume / Momentum"] = (False, "Normal Volume")

        if (aligned_bullish and pcr >= 1.0) or (aligned_bearish and pcr < 1.0):
            score += 25
            checks["Option PCR"] = (True, f"PCR = {pcr}")
        else:
            checks["Option PCR"] = (False, f"PCR = {pcr} (Divergence)")

        # Decision, Strike & Risk-Reward Parameters
        if score >= 70 and aligned_bullish and above_vwap:
            bias = "BUY CALL"
            strike = get_atm_strike(raw_sym, current_price, bias)
            sl = round(current_price * 0.995, 2)
            target1 = round(current_price * 1.008, 2)
            target2 = round(current_price * 1.015, 2)
        elif score >= 70 and aligned_bearish and not above_vwap:
            bias = "BUY PUT"
            strike = get_atm_strike(raw_sym, current_price, bias)
            sl = round(current_price * 1.005, 2)
            target1 = round(current_price * 0.992, 2)
            target2 = round(current_price * 0.985, 2)
        else:
            bias = "NEUTRAL / NO TRADE"
            strike = "N/A"
            sl = 0.0
            target1 = 0.0
            target2 = 0.0

        return {
            "symbol": raw_sym,
            "price": current_price,
            "change_pct": price_change_pct,
            "high": day_high,
            "low": day_low,
            "vwap": current_vwap,
            "pcr": pcr,
            "bias": bias,
            "strike": strike,
            "sl": sl,
            "target1": target1,
            "target2": target2,
            "score": score,
            "checks": checks,
            "trends": {"1D": trend_1d, "1H": trend_1h, "15M": trend_15m, "5M": trend_5m}
        }
    except Exception as e:
        return None

# --- STREAMLIT DASHBOARD UI ---
st.title("⚡ AI Trading Assistant V3")
st.markdown("Live Multi-Timeframe Signals, Strike Selection, & Automated Telegram Dispatcher.")

# Sidebar Controls
st.sidebar.header("⚙️ Configuration")
user_symbol = st.sidebar.text_input("Symbol (e.g. NIFTY, SENSEX, RELIANCE)", value="NIFTY").upper().strip()

st.sidebar.subheader("🔄 Continuous Market Auto-Refresh")
enable_autorefresh = st.sidebar.checkbox("Enable Auto-Refresh", value=False)
refresh_interval = st.sidebar.slider("Refresh Timer (Seconds)", min_value=15, max_value=120, value=30)

if enable_autorefresh:
    st_autorefresh(interval=refresh_interval * 1000, key="market_auto_refresh")

# Interface Tabs
tab_screener, tab_journal = st.tabs(["📊 Live Market Signals", "📓 SQLite Trade Journal"])

with tab_screener:
    data = analyze_market(user_symbol)

    if data:
        # Top Metric Row
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
            pcr_icon = "🟢" if data["pcr"] >= 1.0 else "🔴"
            st.markdown(f"### {pcr_icon} **{data['pcr']}**")

        with col4:
            st.markdown("**Spot Price**")
            st.metric(
                label=f"{data['symbol']}", 
                value=f"₹{data['price']:,.2f}", 
                delta=f"{data['change_pct']:+.2f}%"
            )

        st.divider()

        # Strike Price & Trade Levels Card
        if data["bias"] != "NEUTRAL / NO TRADE":
            st.subheader("🎯 Trade Execution Parameters")
            p1, p2, p3, p4 = st.columns(4)
            p1.markdown(f"**Recommended Strike**\n### `{data['strike']}`")
            p2.markdown(f"**Entry Spot Price**\n### ₹{data['price']:,.2f}")
            p3.markdown(f"**Stop Loss (SL)**\n### ₹{data['sl']:,.2f}")
            p4.markdown(f"**Target 1 / Target 2**\n### ₹{data['target1']:,.2f} / ₹{data['target2']:,.2f}")
            st.divider()

        # Indicators Row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("VWAP", f"₹{data['vwap']:,.2f}")
        m2.metric("Day High", f"₹{data['high']:,.2f}")
        m3.metric("Day Low", f"₹{data['low']:,.2f}")
        m4.metric("1D Trend Direction", data["trends"]["1D"])

        st.divider()

        # Trend Matrix & Checklist
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

        # Action Buttons
        act1, act2 = st.columns(2)

        with act1:
            st.subheader("📲 Telegram Dispatcher")
            alert_msg = (
                f"🚨 *AI TRADING SIGNAL: {data['symbol']}*\n\n"
                f"• *Decision:* `{data['bias']}`\n"
                f"• *Recommended Strike:* `{data['strike']}`\n"
                f"• *Entry Price:* ₹{data['price']:,.2f}\n"
                f"• *Stop Loss:* ₹{data['sl']:,.2f}\n"
                f"• *Target 1:* ₹{data['target1']:,.2f}\n"
                f"• *Confidence Score:* `{data['score']}%`\n"
                f"• *PCR Ratio:* `{data['pcr']}`\n\n"
                f"⏰ _Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
            )
            st.text_area("Telegram Preview", value=alert_msg, height=140)
            if st.button("🚀 Send Alert to Telegram"):
                if send_telegram_alert(alert_msg):
                    st.success("✅ Signal broadcasted to Telegram successfully!")

        with act2:
            st.subheader("📓 Journal Entry Logging")
            notes = st.text_input("Custom Notes", value=f"{data['bias']} setup logged for {data['symbol']}")
            if st.button("💾 Log Trade to SQLite Database"):
                log_trade(
                    symbol=data['symbol'],
                    price=data['price'],
                    bias=data['bias'],
                    strike=data['strike'],
                    entry=data['price'],
                    sl=data['sl'],
                    target=data['target1'],
                    score=data['score'],
                    pcr=data['pcr'],
                    notes=notes
                )
                st.success(f"✅ Trade recorded in SQLite (`{DB_NAME}`)!")

    else:
        st.warning(f"Unable to fetch data for '{user_symbol}'. Try typing NIFTY, BANKNIFTY, SENSEX, or stock tickers like RELIANCE, TCS, INFY.")

with tab_journal:
    st.subheader("📓 SQLite Automated Trade Journal History")
    journal_df = get_journal_data()
    
    if not journal_df.empty:
        st.dataframe(journal_df, use_container_width=True)
        st.caption(f"Total Entries Logged: {len(journal_df)}")
    else:
        st.info("No trade entries logged yet. Log your first setup from the Live Market Signals tab!")
