import streamlit as st
import requests
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="Trading Assistant", page_icon="📈", layout="wide")

# --- TELEGRAM FUNCTION ---
def send_telegram_alert(message: str) -> bool:
    """Sends a notification message to Telegram."""
    bot_token = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = st.secrets.get("TELEGRAM_CHAT_ID", "")

    if not bot_token or not chat_id:
        st.error("⚠️ Telegram credentials missing in Streamlit Secrets!")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        res_data = response.json()
        if not res_data.get("ok"):
            st.error(f"Telegram API Error: {res_data.get('description')}")
            return False
        return True
    except Exception as e:
        st.error(f"Failed to connect to Telegram: {e}")
        return False

# --- LIVE STOCK DATA FUNCTION ---
@st.cache_data(ttl=15)
def get_stock_data(symbol: str):
    """Fetches real-time stock data from Yahoo Finance for Indian stocks."""
    ticker_symbol = symbol.upper().strip()
    if not ticker_symbol.endswith(".NS") and not ticker_symbol.endswith(".BO"):
        ticker_symbol += ".NS"  # Default to NSE
        
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.fast_info
        last_price = info.last_price
        prev_close = info.previous_close
        
        if last_price is None:
            return None
            
        change = last_price - prev_close
        p_change = (change / prev_close) * 100
        
        return {
            "symbol": symbol.upper(),
            "price": last_price,
            "change": change,
            "p_change": p_change,
            "high": info.day_high,
            "low": info.day_low
        }
    except Exception as e:
        return None

# --- UI INTERFACE ---
st.title("📈 Live Trading Assistant")
st.markdown("Real-time NSE stock data & automated Telegram alerting.")

# Sidebar
st.sidebar.header("Configuration")
user_symbol = st.sidebar.text_input("NSE Stock Symbol", value="RELIANCE").upper().strip()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Live Market Data: `{user_symbol}`")
    
    data = get_stock_data(user_symbol)

    if data:
        last_price = data["price"]
        
        # Display Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Last Price", f"₹{last_price:,.2f}", f"{data['change']:+.2f} ({data['p_change']:+.2f}%)")
        m2.metric("Day High", f"₹{data['high']:,.2f}" if data['high'] else "N/A")
        m3.metric("Day Low", f"₹{data['low']:,.2f}" if data['low'] else "N/A")
    else:
        last_price = None
        st.warning(f"Could not fetch data for symbol '{user_symbol}'. Ensure it is a valid NSE stock ticker (e.g., RELIANCE, TCS, INFY).")

with col2:
    st.subheader("📲 Send Manual Alert")
    
    default_msg = f"Update: {user_symbol} is currently trading at ₹{last_price:,.2f}" if last_price else f"Update on {user_symbol}"
    custom_msg = st.text_area("Custom Telegram Message", value=default_msg)
    
    if st.button("Send to Telegram"):
        if send_telegram_alert(custom_msg):
            st.success("✅ Telegram notification delivered successfully!")
