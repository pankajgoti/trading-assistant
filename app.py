import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import sqlite3
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# Page Configuration
st.set_page_config(
    page_title="AI Trading Assistant V2",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1E222D;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #2A2E39;
        text-align: center;
    }
    .buy-call { color: #00E676; font-size: 28px; font-weight: bold; }
    .buy-put { color: #FF5252; font-size: 28px; font-weight: bold; }
    .neutral { color: #FFD600; font-size: 28px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- 1. SQLITE DATABASE SETUP ---
DB_NAME = "trade_journal.db"

def init_db():
    """Initializes SQLite database for automated trade journaling."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            price REAL,
            bias TEXT,
            score INTEGER,
            vwap REAL,
            pcr REAL,
            notes TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_trade(symbol, price, bias, score, vwap, pcr, notes="Auto-Logged"):
    """Saves trade entry into SQLite database."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        INSERT INTO journal (timestamp, symbol, price, bias, score, vwap, pcr, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), symbol, price, bias, score, vwap, pcr, notes))
    conn.commit()
    conn.close()

def get_journal_data():
    """Fetches all trade entries from SQLite."""
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM journal ORDER BY id DESC", conn)
    conn.close()
    return df

init_db()

# --- 2. TELEGRAM NOTIFICATION FUNCTION ---
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
        st.error(f"Telegram Delivery Failed: {e}")
        return False

# --- 3. TECHNICAL ANALYSIS & OPTION CHAIN HELPERS ---
def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculates Volume Weighted Average Price (VWAP)."""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    return (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()

def get_option_chain_pcr(ticker_obj) -> float:
    """Calculates Put-Call Ratio (PCR) from option open interest."""
    try:
        expirations = ticker_obj.options
        if not expirations:
            return 1.0  # Default neutral PCR
        
        # Use nearest expiration cycle
        chain = ticker_obj.option_chain(expirations[0])
        total_call_oi = chain.calls['openInterest'].sum()
        total_put_oi = chain.puts['openInterest'].sum()
        
        if total_call_oi > 0:
            return round(total_put_oi / total_call_oi, 2)
        return 1.0
    except Exception:
        return 1.0

@st.cache_data(ttl=20)
def analyze_stock(symbol: str):
    """Fetches multi-timeframe data, Option Chain, & runs signal logic."""
    ticker_sym = symbol.upper().strip()
    if not ticker_sym.endswith(".NS") and not ticker_sym.startswith("^"):
        ticker_sym += ".NS"

    try:
        ticker = yf.Ticker(ticker_sym)
        
        # Fetch Multi-Timeframe Data
        df_5m = ticker.history(period="5d", interval="5m")
        df_15m = ticker.history(period="10d", interval="15m")
        df_1h = ticker.history(period="1mo", interval="1h")
        df_1d = ticker.history(period="6mo", interval="1d")

        if df_5m.empty or len(df_5m) < 20:
            return None

        # Current Price Metrics
        current_price = df_5m['Close'].iloc[-1]
        prev_close = df_1d['Close'].iloc[-2] if len(df_1d) > 1 else current_price
        day_high = df_5m['High'].max()
        day_low = df_5m['Low'].min()
        price_change_pct = ((current_price - prev_close) / prev_close) * 100

        # Multi-Timeframe Trend
        trend_1d = "BULLISH" if df_1d['Close'].iloc[-1] > df_1d['Close'].rolling(20).mean().iloc[-1] else "BEARISH"
        trend_1h = "BULLISH" if df_1h['Close'].iloc[-1] > df_1h['Close'].rolling(20).mean().iloc[-1] else "BEARISH"
        trend_15m = "BULLISH" if df_15m['Close'].iloc[-1] > df_15m['Close'].rolling(20).mean().iloc[-1] else "BEARISH"
        trend_5m = "BULLISH" if df_5m['Close'].iloc[-1] > df_5m['Close'].rolling(20).mean().iloc[-1] else "BEARISH"

        # VWAP & Volume
        df_5m['VWAP'] = calculate_vwap(df_5m)
        current_vwap = df_5m['VWAP'].iloc[-1]
        above_vwap = current_price > current_vwap

        recent_vol = df_5m['Volume'].iloc[-1]
        avg_vol = df_5m['Volume'].rolling(20).mean().iloc[-1]
        high_volume = recent_vol > (1.5 * avg_vol)

        # Option Chain Put-Call Ratio
        pcr = get_option_chain_pcr(ticker)

        # Confidence Score Calculation
        score = 0
        checks = {}

        # Rule 1: Multi-timeframe Alignment (+30 pts)
        aligned_bullish = (trend_1d == "BULLISH" and trend_1h == "BULLISH" and trend_15m == "BULLISH")
        aligned_bearish = (trend_1d == "BEARISH" and trend_1h == "BEARISH" and trend_15m == "BEARISH")
        
        if aligned_bullish or aligned_bearish:
            score += 30
            checks["MTF Trend"] = (True, "1D, 1H, 15M Aligned")
        else:
            checks["MTF Trend"] = (False, "Mixed Timeframe Signals")

        # Rule 2: VWAP Position (+25 pts)
        if (aligned_bullish and above_vwap) or (aligned_bearish and not above_vwap):
            score += 25
            checks["VWAP"] = (True, f"{'Above' if above_vwap else 'Below'} VWAP")
        else:
            checks["VWAP"] = (False, "Conflicting VWAP Position")

        # Rule 3: Volume Surge (+20 pts)
        if high_volume:
            score += 20
            checks["Volume"] = (True, f"Surge ({recent_vol:,.0f})")
        else:
            checks["Volume"] = (False, "Normal Volume")

        # Rule 4: Option Chain PCR Sentiment (+25 pts)
        if (aligned_bullish and pcr >= 1.0) or (aligned_bearish and pcr < 1.0):
            score += 25
            checks["Option PCR"] = (True, f"PCR = {pcr} (Confirms sentiment)")
        else:
            checks["Option PCR"] = (False, f"PCR = {pcr} (Divergence)")

        # Decision Logic
        if score >= 70 and aligned_bullish and above_vwap:
            bias = "BUY CALL"
        elif score >= 70 and aligned_bearish and not above_vwap:
            bias = "BUY PUT"
        else:
            bias = "NEUTRAL / NO TRADE"

        return {
            "symbol": symbol.upper(),
            "price": current_price,
            "change_pct": price_change_pct,
            "high": day_high,
            "low": day_low,
            "vwap": current_vwap,
            "pcr": pcr,
            "bias": bias,
            "score": score,
            "checks": checks,
            "trends": {"1D": trend_1d, "1H": trend_1h, "15M": trend_15m, "5M": trend_5m}
        }
    except Exception as e:
        return None

# --- STREAMLIT UI ---
st.title("⚡ AI Trading Assistant V2")
st.markdown("Live Multi-Timeframe Screening, Option Chain PCR, & Automated SQLite Journaling.")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")
user_symbol = st.sidebar.text_input("Stock / Index Symbol", value="RELIANCE").upper().strip()

# --- AUTO-REFRESH TIMER ---
st.sidebar.subheader("🔄 Market Hours Auto-Refresh")
enable_autorefresh = st.sidebar.checkbox("Enable Live Auto-Refresh", value=False)
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", min_value=15, max_value=120, value=30)

if enable_autorefresh:
    st_autorefresh(interval=refresh_interval * 1000, key="market_screener_autorefresh")

# Main Navigation Tabs
tab_screener, tab_journal = st.tabs(["📊 Live Screener & Signals", "📓 Trade Journal (SQLite)"])

with tab_screener:
    data = analyze_stock(user_symbol)

    if data:
        # Metrics Header
        col1, col2, col3, col4 = st.columns([1.5, 1.2, 1.2, 1.5])
        
        with col1:
            st.markdown("**Decision**")
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
            pcr_color = "🟢" if data["pcr"] >= 1.0 else "🔴"
            st.markdown(f"### {pcr_color} **{data['pcr']}**")

        with col4:
            st.markdown("**Live Price**")
            st.metric(
                label=f"{data['symbol']} (NSE)", 
                value=f"₹{data['price']:,.2f}", 
                delta=f"{data['change_pct']:+.2f}%"
            )

        st.divider()

        # Indicators Bar
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("VWAP", f"₹{data['vwap']:,.2f}")
        m2.metric("Day High", f"₹{data['high']:,.2f}")
        m3.metric("Day Low", f"₹{data['low']:,.2f}")
        m4.metric("1D Trend", data["trends"]["1D"])

        st.divider()

        # Multi-Timeframe Alignment & Checklist
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("📊 Multi-Timeframe Alignment")
            t_df = pd.DataFrame([data["trends"]]).T
            t_df.columns = ["Trend Direction"]
            st.table(t_df)

        with c2:
            st.subheader("🎯 Signal Checklist")
            for check, (passed, desc) in data["checks"].items():
                icon = "🟢" if passed else "🔴"
                st.write(f"{icon} **{check}**: {desc}")

        st.divider()

        # Actions: Telegram Alert & SQLite Logging
        act1, act2 = st.columns(2)

        with act1:
            st.subheader("📲 Telegram Dispatcher")
            alert_msg = (
                f"🚨 *AI TRADING SIGNAL: {data['symbol']}*\n\n"
                f"• *Decision:* `{data['bias']}`\n"
                f"• *Confidence:* `{data['score']}%`\n"
                f"• *Current Price:* ₹{data['price']:,.2f}\n"
                f"• *VWAP:* ₹{data['vwap']:,.2f}\n"
                f"• *PCR:* `{data['pcr']}`\n"
                f"• *MTF Trends:* `{data['trends']['1D']} / {data['trends']['1H']} / {data['trends']['15M']}`\n\n"
                f"⏰ _Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
            )
            st.text_area("Alert Preview", value=alert_msg, height=120)
            if st.button("🚀 Send Signal to Telegram"):
                if send_telegram_alert(alert_msg):
                    st.success("✅ Telegram signal sent!")

        with act2:
            st.subheader("📓 Journal Entry Logging")
            notes = st.text_input("Trade Notes", value=f"{data['bias']} signal setup for {data['symbol']}")
            if st.button("💾 Log Signal to SQLite Database"):
                log_trade(
                    symbol=data['symbol'],
                    price=data['price'],
                    bias=data['bias'],
                    score=data['score'],
                    vwap=data['vwap'],
                    pcr=data['pcr'],
                    notes=notes
                )
                st.success(f"✅ Successfully recorded trade in `{DB_NAME}`!")

    else:
        st.warning(f"Unable to analyze '{user_symbol}'. Please verify the stock ticker symbol (e.g., RELIANCE, INFY, TATAMOTORS).")

with tab_journal:
    st.subheader("📓 SQLite Automated Trade Journal History")
    journal_df = get_journal_data()
    
    if not journal_df.empty:
        st.dataframe(journal_df, use_container_width=True)
        st.caption(f"Total Logged Trades: {len(journal_df)}")
    else:
        st.info("No trade entries logged yet. Log your first signal from the Live Screener tab!")
