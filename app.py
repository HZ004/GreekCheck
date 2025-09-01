import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import threading
import time
from upstox_api.api import Upstox

# =========================
# GLOBAL CONFIG
# =========================
st.set_page_config(page_title="Live Option Greeks Dashboard", layout="wide")
API_KEY = "YOUR_UPSTOX_API_KEY"  # Replace with your API Key
FEED_INTERVAL = 1  # seconds between data refresh
SELECTED_CONTRACTS = []
EXPIRY_LIST = []
STOP_FETCHING = False


# =========================
# FUNCTION: INIT UPSTOX
# =========================
def init_upstox(access_token):
    u = Upstox(API_KEY, access_token)
    u.get_master_contract('NSE_EQ')  # Load master contracts
    u.get_master_contract('NSE_FO')
    return u


# =========================
# FUNCTION: FETCH EXPIRIES
# =========================
def get_expiries(upstox_obj, symbol="NIFTY"):
    contracts = upstox_obj.get_master_contract('NSE_FO')
    expiries = sorted({c.expiry for c in contracts.values() if c.symbol == symbol})
    return expiries[:4]  # return next 4 expiries


# =========================
# FUNCTION: GET ATM STRIKE
# =========================
def get_atm_strike(upstox_obj, symbol="NIFTY"):
    quote = upstox_obj.get_live_feed(f"NSE_EQ|{symbol}", "ltp")
    ltp = quote['ltp']
    atm = round(ltp / 50) * 50  # Assuming NIFTY strike interval of 50
    return atm


# =========================
# FUNCTION: SELECT CONTRACTS
# =========================
def select_contracts(upstox_obj, symbol="NIFTY", expiry=None):
    global SELECTED_CONTRACTS
    atm = get_atm_strike(upstox_obj, symbol)
    strikes = [atm + i * 50 for i in range(-5, 6)]  # 5 ITM + 5 OTM
    contracts = upstox_obj.get_master_contract('NSE_FO')
    selected = [
        c for c in contracts.values()
        if c.symbol == symbol and c.expiry == expiry and c.strike_price in strikes
    ]
    SELECTED_CONTRACTS = selected


# =========================
# FUNCTION: FETCH GREEKS DATA
# =========================
def fetch_greeks(upstox_obj):
    global STOP_FETCHING
    while not STOP_FETCHING:
        now = dt.datetime.now().time()
        if dt.time(9, 20) <= now <= dt.time(15, 20):
            data = []
            for c in SELECTED_CONTRACTS:
                try:
                    q = upstox_obj.get_live_feed(f"{c.exchange}|{c.token}", "full")
                    data.append({
                        "Symbol": c.symbol,
                        "Strike": c.strike_price,
                        "Type": c.instrument_type,
                        "LTP": q.get('ltp', np.nan),
                        "IV": q.get('impliedVolatility', np.nan),
                        "Delta": q.get('delta', np.nan),
                        "Gamma": q.get('gamma', np.nan),
                        "Theta": q.get('theta', np.nan),
                        "Vega": q.get('vega', np.nan)
                    })
                except Exception as e:
                    print("Error fetching:", e)
            if data:
                st.session_state["greeks_df"] = pd.DataFrame(data)
        time.sleep(FEED_INTERVAL)


# =========================
# STREAMLIT UI
# =========================
st.title("📈 Live NIFTY Option Greeks Dashboard")
token = st.text_input("Enter your Upstox Access Token:", type="password")

if token:
    # Initialize Upstox
    u = init_upstox(token)

    # Fetch expiries
    if not EXPIRY_LIST:
        EXPIRY_LIST = get_expiries(u)
        st.session_state["selected_expiry"] = EXPIRY_LIST[0]

    # Expiry selection
    expiry = st.selectbox("Select Expiry", EXPIRY_LIST, index=0)
    st.session_state["selected_expiry"] = expiry

    # Auto-select contracts at 9:16 AM daily
    now = dt.datetime.now().time()
    if dt.time(9, 16) <= now <= dt.time(9, 20):
        select_contracts(u, expiry=expiry)

    # Start background thread
    if "fetch_thread" not in st.session_state:
        st.session_state["fetch_thread"] = threading.Thread(target=fetch_greeks, args=(u,), daemon=True)
        st.session_state["fetch_thread"].start()

    # Display live data
    placeholder = st.empty()
    while True:
        if "greeks_df" in st.session_state:
            df = st.session_state["greeks_df"]
            placeholder.dataframe(df, use_container_width=True)
        time.sleep(1)
