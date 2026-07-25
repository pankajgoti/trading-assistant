import streamlit as st
import pandas as pd
import numpy as np
import datetime
import requests

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Trading Assistant & Screener",
    page_icon="📈",
    layout="wide"
)

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #1e2530;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2d3748;
        text-align: center;
    }
    .bullish { color: #2ecc71; font-weight: bold; }
    .bearish { color: #e74c3c; font-weight: bold; }
    .neutral { color: #f1c40f; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Control Panel")
selected_index = st.sidebar.selectbox("Select Index / Asset", ["NIFTY 50", "BANK NIFTY", "FINNIFTY"])
timeframe = st.sidebar.selectbox("Primary Timeframe", ["5m", "15m", "1h", "Daily"])
confidence_threshold = st.sidebar.slider("Min Confidence Threshold for Alert (%)", 50, 95, 80)

st.sidebar.markdown("---")
st.sidebar.subheader("📲 Telegram Bot Setup")
tg_token = st.sidebar.text_input("Bot Token", type="password")
tg_chat_id = st.sidebar.text_input("Chat ID", type="password")

# --- MAIN DASHBOARD HEADER ---
st.title(f"🤖 AI Trading Assistant: {selected_index}")
st.markdown(f"**Current Time:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **Mode:** Live Simulation")

# --- MOCK DATA ENGINE (Replace with live broker/market feeds later) ---
# In production, compute these dynamically using pandas/ta on live historical data.
market_status = {
    "Trend": ("Bullish", "🟢"),
    "Strength": ("Strong", "🟢"),
    "Momentum": ("High", "🟢"),
    "VWAP": ("Above", "🟢"),
    "ORB": ("Breakout", "🟢"),
    "Volume": ("High", "🟢"),
    "OI Build-up": ("Call Writing / Long", "🟢"),
    "India VIX": ("13.4 (Moderate)", "🟡")
}

confidence_scores = {
    "Trend Alignment (Multi-TF)": {"score": 20, "max": 20},
    "Volume Expansion": {"score": 15, "max": 20},
    "VWAP Positioning": {"score": 10, "max": 10},
    "Options Data (OI/PCR)": {"score": 15, "max": 20},
    "Smart Money Concepts (SMC)": {"score": 12, "max": 15},
    "India VIX & Regime": {"score": 10, "max": 10},
    "Market Breadth": {"score": 8, "max": 10},
}

total_confidence = sum([item["score"] for item in confidence_scores.values()])

# --- LAYOUT: TWO COLUMNS (DASHBOARD & DECISION PANEL) ---
col1, col2 = st.run_cols = st.columns([1.2, 1]) if hasattr(st, 'columns') else st.columns(2)

with col1:
    st.subheader("📊 Market Internal Matrix")
    
    # Display Status Grid
    status_df = pd.DataFrame(
        [[v[1], k, v[0]] for k, v in market_status.items()],
        columns=["Status", "Indicator", "Condition"]
    )
    st.table(status_df)
    
    st.subheader("🔍 AI Confidence Score Breakdown")
    score_data = []
    for k, v in confidence_scores.items():
        pct = int((v["score"] / v["max"]) * 100)
        score_data.append({"Component": k, "Score": f"{v['score']}/{v['max']}", "Health": f"{pct}%"})
    st.dataframe(pd.DataFrame(score_data), hide_index=True, use_container_width=True)

with col2:
    st.subheader("🎯 Automated Decision Hub")
    
    if total_confidence >= confidence_threshold:
        st.markdown("""
        <div style="background-color: #143d29; padding: 20px; border-radius: 10px; border: 1px solid #2ecc71;">
            <h2 style="color: #2ecc71; margin:0;">DECISION: BUY CALL 🟢</h2>
            <p style="font-size: 18px; margin-top: 5px;"><b>Confidence Score:</b> {}%</p>
        </div>
        """.format(total_confidence), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: #4a2c2a; padding: 20px; border-radius: 10px; border: 1px solid #e74c3c;">
            <h2 style="color: #e74c3c; margin:0;">DECISION: NO TRADE 🔴</h2>
            <p style="font-size: 18px; margin-top: 5px;"><b>Confidence Score:</b> {}% (Below threshold)</p>
        </div>
        """.format(total_confidence), unsafe_allow_html=True)
        
    st.markdown("### Trade Parameters (Setup #104)")
    param_col1, param_col2 = st.columns(2)
    with param_col1:
        st.metric(label="Recommended Strike", value="25200 CE")
        st.metric(label="Entry Price", value="₹122.00")
        st.metric(label="Target 1", value="₹145.00")
    with param_col2:
        st.metric(label="Stop Loss", value="₹108.00")
        st.metric(label="Target 2", value="₹168.00")
        st.metric(label="Target 3", value="Trail S/L")

    st.markdown("#### 💡 AI Rationale")
    st.markdown("""
    * ✓ Price is cleanly sustained **above VWAP** on the 15m/5m charts.
    * ✓ Volume nodes indicate accumulation near the Point of Control (POC).
    * ✓ PCR shifted bullish with aggressive put writing at the 25200 strike.
    * ✓ India VIX is steady, favoring directional trend day rules.
    """)

    # Telegram Alert Button
    if st.button("🚀 Push Alert to Telegram Now"):
        if tg_token and tg_chat_id:
            message = f"🚨 *AI TRADING ALERT: {selected_index}* 🚨\n\nDecision: **BUY CALL**\nConfidence: **{total_confidence}%**\nStrike: 25200 CE\nEntry: ₹122\nStop Loss: ₹108\nTargets: ₹145 / ₹168"
            url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
            payload = {"chat_id": tg_chat_id, "text": message, "parse_mode": "Markdown"}
            try:
                response = requests.post(url, json=payload)
                if response.status_code == 200:
                    st.success("Alert successfully broadcasted to Telegram!")
                else:
                    st.error(f"Failed to send. Error code: {response.status_code}")
            except Exception as e:
                st.error(f"Connection error: {e}")
        else:
            st.warning("Please input your Telegram Bot Token and Chat ID in the sidebar first.")

# --- AUTOMATED JOURNAL SECTION ---
st.markdown("---")
st.subheader("📓 Automated Trade Journal Log")
journal_data = pd.DataFrame([
    {"Date": "2026-06-24", "Index": "NIFTY", "Type": "CE", "Entry": 110, "Exit": 142, "P&L (+/-)": "+₹1,600", "Result": "Win"},
    {"Date": "2026-06-23", "Index": "BANK NIFTY", "Type": "PE", "Entry": 310, "Exit": 290, "P&L (+/-)": "-₹1,000", "Result": "Loss"},
    {"Date": "2026-06-22", "Index": "NIFTY", "Type": "CE", "Entry": 95, "Exit": 130, "P&L (+/-)": "+₹1,750", "Result": "Win"},
])
st.dataframe(journal_data, hide_index=True, use_container_width=True)
