import streamlit as st
import requests
import pandas as pd
from nsepython import nse_eq

# Page Configuration
st.set_page_config(page_title="Trading Assistant", page_icon="📈", layout="wide")

# --- 1. TELEGRAM BOT INTEGRATION ---
def send_telegram_alert(message: str) -> bool:
    """Sends a notification message to Telegram."""
    bot_token = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = st.secrets.get("TELEGRAM_CHAT_ID", "")

    if not bot_token or not chat_id:
        st.error("Telegram credentials missing in Streamlit Secrets.")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        return response.json().get("ok", False)
    except Exception as e:
        st.error(f"Failed to send Telegram alert: {e}")
        return False

# --- 2. LIVE NSE DATA FETCHING ---
@st.cache_data(ttl=30)  # Caches result for 30 seconds to stay within NSE rate limits
def fetch_nse_live_quote(symbol: str):
    """Fetches real-time equity data directly from NSE."""
    try:
        data = nse_eq(symbol)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

# --- STREAMLIT USER INTERFACE ---
st.title("📈 Live Trading Assistant")
st.markdown("Real-time NSE stock data & automated Telegram alerting.")

# Sidebar Controls
st.sidebar.header("Configuration")
symbol = st.sidebar.text_input("NSE Stock Symbol", value="RELIANCE").upper().strip()
refresh = st.sidebar.button("🔄 Refresh Data")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Live Market Data: `{symbol}`")
    
    with st.spinner("Fetching latest price from NSE..."):
        stock_data = fetch_nse_live_quote(symbol)

    if stock_data and "priceInfo" in stock_data:
        price_info = stock_data["priceInfo"]
        last_price = price_info.get("lastPrice", 0.0)
        change = price_info.get("change", 0.0)
        p_change = price_info.get("pChange", 0.0)
        
        # Display Key Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Last Traded Price", f"₹{last_price:,.2f}", f"{change:+.2f} ({p_change:+.2f}%)")
        m2.metric("Day High", f"₹{price_info.get('intraDayHighLow', {}).get('max', 0):,.2f}")
        m3.metric("Day Low", f"₹{price_info.get('intraDayHighLow', {}).get('min', 0):,.2f}")

        # Price Alert Condition Check
        alert_threshold = st.number_input("Set Price Alert Threshold (₹):", value=float(last_price))
        if st.button("Trigger Alert Evaluation"):
            if last_price >= alert_threshold:
                alert_text = f"🚨 *PRICE ALERT*\n\n*{symbol}* has crossed your target price!\nCurrent Price: ₹{last_price}\nTarget: ₹{alert_threshold}"
                if send_telegram_alert(alert_text):
                    st.success("Alert sent to Telegram!")
            else:
                st.info(f"Current price (₹{last_price}) has not reached target (₹{alert_threshold}).")

with col2:
    st.subheader("📲 Send Manual Alert")
    custom_msg = st.text_area("Custom Telegram Message", value=f"Update: {symbol} is currently trading at ₹{last_price if stock_data else 'N/A'}")
    
    if st.button("Send to Telegram"):
        if send_telegram_alert(custom_msg):
            st.success("Telegram notification delivered successfully!")
        else:
            st.error("Could not send Telegram notification.")
