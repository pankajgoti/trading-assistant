import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="AI Trading Assistant V1",
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

# --- TELEGRAM NOTIFICATION FUNCTION ---
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

# --- TECHNICAL ANALYSIS HELPERS ---
def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculates Volume Weighted Average Price (VWAP)."""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    return (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()

@st.cache_data(ttl=30)
def analyze_stock(symbol: str):
    """Fetches multi-timeframe data & runs signal logic."""
    ticker_sym = symbol.upper().strip()
    if not ticker_sym.endswith(".NS") and not ticker_sym.startswith("^"):
        ticker_sym += ".NS"

    try:
        ticker = yf.Ticker(ticker_sym)
        
        # 1. Fetch Multi-Timeframe Data
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

        # --- 2. MULTI-TIMEFRAME TREND CHECK ---
        trend_1d = "BULLISH" if df_1d['Close'].iloc[-1] > df_1d['Close'].rolling(20).mean().iloc[-1] else "BEARISH"
        trend_1h = "BULLISH" if df_1h['Close'].iloc[-1] > df_1h['Close'].rolling(20).mean().iloc[-1] else "BEARISH"
        trend_15m = "BULLISH" if df_15m['Close'].iloc[-1] > df_15m['Close'].rolling(20).mean().iloc[-1] else "BEARISH"
        trend_5m = "BULLISH" if df_5m['Close'].iloc[-1] > df_5m['Close'].rolling(20).mean().iloc[-1] else "BEARISH"

        # --- 3. VWAP & VOLUME SURGE ANALYSIS ---
        df_5m['VWAP'] = calculate_vwap(df_5m)
        current_vwap = df_5m['VWAP'].iloc[-1]
        above_vwap = current_price > current_vwap

        recent_vol = df_5m['Volume'].iloc[-1]
        avg_vol = df_5m['Volume'].rolling(20).mean().iloc[-1]
        high_volume = recent_vol > (1.5 * avg_vol)

        # --- 4. CONFIDENCE SCORE ALGORITHM ---
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
            checks["Volume"] = (True, f"Surge ({recent_vol:,.0f} vs avg {avg_vol:,.0f})")
        else:
            checks["Volume"] = (False, "Normal Volume")

        # Rule 4: Momentum Confirmation (+25 pts)
        if (aligned_bullish and trend_5m == "BULLISH") or (aligned_bearish and trend_5m == "BEARISH"):
            score += 25
            checks["5M Trigger"] = (True, "Short-term momentum aligned")
        else:
            checks["5M Trigger"] = (False, "5M Momentum lagging")

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
            "bias": bias,
            "score": score,
            "checks": checks,
            "trends": {"1D": trend_1d, "1H": trend_1h, "15M": trend_15m, "5M": trend_5m}
        }
    except Exception as e:
        return None

# --- UI DASHBOARD ---
st.title("⚡ AI Trading Assistant & Screener")
st.markdown("Multi-timeframe trend alignment, VWAP positioning & automated Telegram alerts.")

# Sidebar Configuration
st.sidebar.header("Configuration")
user_symbol = st.sidebar.text_input("Stock / Index Symbol", value="RELIANCE").upper().strip()
min_confidence = st.sidebar.slider("Min Alert Confidence Score (%)", min_value=50, max_value=90, value=70)

if st.sidebar.button("🔄 Refresh Analysis"):
    st.cache_data.clear()

# Fetch Analysis Data
data = analyze_stock(user_symbol)

if data:
    # Top Row: Signal & Confidence
    col1, col2, col3 = st.columns([1.5, 1.5, 2])
    
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
        st.markdown("**Live Price**")
        st.metric(
            label=f"{data['symbol']} (NSE)", 
            value=f"₹{data['price']:,.2f}", 
            delta=f"{data['change_pct']:+.2f}%"
        )

    st.divider()

    # Second Row: Market Internals & Indicators
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("VWAP", f"₹{data['vwap']:,.2f}")
    m2.metric("Day High", f"₹{data['high']:,.2f}")
    m3.metric("Day Low", f"₹{data['low']:,.2f}")
    m4.metric("1D Trend", data["trends"]["1D"])

    st.divider()

    # Third Row: Multi-Timeframe Matrix & Checkpoints
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

    # Fourth Row: Telegram Alert Trigger
    st.subheader("📲 Telegram Signal Dispatcher")
    
    alert_msg = (
        f"🚨 *AI TRADING SIGNAL: {data['symbol']}*\n\n"
        f"• *Decision:* `{data['bias']}`\n"
        f"• *Confidence:* `{data['score']}%`\n"
        f"• *Current Price:* ₹{data['price']:,.2f}\n"
        f"• *VWAP:* ₹{data['vwap']:,.2f}\n"
        f"• *1D / 1H / 15M:* `{data['trends']['1D']} / {data['trends']['1H']} / {data['trends']['15M']}`\n\n"
        f"⏰ _Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
    )

    st.text_area("Alert Preview", value=alert_msg, height=130)

    if st.button("🚀 Send Signal to Telegram"):
        if send_telegram_alert(alert_msg):
            st.success("✅ Telegram signal sent successfully!")
        else:
            st.error("Failed to deliver signal. Check your Secrets configuration.")

else:
    st.warning(f"Unable to analyze '{user_symbol}'. Please verify the stock ticker symbol (e.g., RELIANCE, INFY, TATAMOTORS).")
