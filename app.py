import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import websocket
import json
import threading
import datetime as dt
from dateutil import parser
from scipy.stats import norm

# ========================
# Black-Scholes Greeks
# ========================
def bs_greeks(S, K, T, r, sigma, option_type="call"):
    """Return delta, gamma, theta, vega for an option."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type == "call":
        theta -= r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        theta += r * K * np.exp(-r * T) * norm.cdf(-d2)
    return delta, gamma, theta, vega

# ========================
# Globals
# ========================
LIVE_DATA = {}
SELECTED_CONTRACTS = []

# ========================
# Fetch Instruments
# ========================
def fetch_instruments(token):
    url = "https://api-v2.upstox.com/instruments"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    data = response.json()
    df = pd.DataFrame(data["data"])
    return df

# ========================
# Get ATM strikes
# ========================
def get_contracts(df, spot, expiry, symbol="NIFTY"):
    opt_df = df[(df["instrument_type"].isin(["CE", "PE"])) & (df["name"] == symbol) & (df["expiry"] == expiry)]
    strikes = sorted(opt_df["strike"].unique())
    atm_strike = min(strikes, key=lambda x: abs(x - spot))
    idx = strikes.index(atm_strike)
    selected_strikes = strikes[max(0, idx-5):idx+6]  # 5 OTM + 5 ITM
    final_df = opt_df[opt_df["strike"].isin(selected_strikes)]
    return final_df

# ========================
# WebSocket Handler
# ========================
def on_message(ws, message):
    data = json.loads(message)
    if "feeds" in data:
        for key, val in data["feeds"].items():
            price = val.get("ltp")
            if price:
                LIVE_DATA[key] = price

def start_websocket(token, instrument_keys):
    def run():
        ws = websocket.WebSocketApp(
            "wss://api-v2.upstox.com/feed/market-data-feed",
            header={"Authorization": f"Bearer {token}"},
            on_message=on_message
        )
        ws.on_open = lambda ws: ws.send(json.dumps({
            "guid": "main",
            "method": "sub",
            "data": {"instrumentKeys": instrument_keys, "mode": "full"}
        }))
        ws.run_forever()
    thread = threading.Thread(target=run)
    thread.start()

# ========================
# Streamlit UI
# ========================
st.title("📈 NIFTY Options Live Greeks Dashboard")

access_token = st.text_input("Enter Upstox Access Token:", type="password")

if access_token:
    st.success("Access Token Set!")
    if st.button("Initialize Contracts"):
        st.info("Fetching instruments...")
        inst_df = fetch_instruments(access_token)
        spot = float(inst_df[(inst_df["instrument_type"] == "EQ") & (inst_df["name"] == "NIFTY")]["close"].iloc[0])
        expiry = sorted(inst_df[inst_df["name"] == "NIFTY"]["expiry"].unique())[0]
        contracts_df = get_contracts(inst_df, spot, expiry)
        global SELECTED_CONTRACTS
        SELECTED_CONTRACTS = contracts_df.to_dict("records")
        keys = contracts_df["instrument_key"].tolist()
        start_websocket(access_token, keys)
        st.success(f"Tracking {len(keys)} contracts")

if SELECTED_CONTRACTS:
    st.subheader("Live Greeks")
    greek_data = []
    now = dt.datetime.now().time()
    if dt.time(9,20) <= now <= dt.time(15,20):
        for c in SELECTED_CONTRACTS:
            symbol = c["tradingsymbol"]
            price = LIVE_DATA.get(c["instrument_key"], c["close"])
            S = price
            K = c["strike"]
            expiry_date = parser.parse(c["expiry"])
            T = max((expiry_date - dt.datetime.now()).days / 365.0, 0.0001)
            sigma = 0.2  # Assume fixed volatility
            r = 0.05
            opt_type = "call" if c["instrument_type"] == "CE" else "put"
            delta, gamma, theta, vega = bs_greeks(S, K, T, r, sigma, opt_type)
            greek_data.append([symbol, price, delta, gamma, theta, vega])
        df = pd.DataFrame(greek_data, columns=["Symbol", "LTP", "Delta", "Gamma", "Theta", "Vega"])
        st.dataframe(df)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Symbol"], y=df["Delta"], mode="lines+markers", name="Delta"))
        fig.add_trace(go.Scatter(x=df["Symbol"], y=df["Gamma"], mode="lines+markers", name="Gamma"))
        fig.add_trace(go.Scatter(x=df["Symbol"], y=df["Theta"], mode="lines+markers", name="Theta"))
        fig.add_trace(go.Scatter(x=df["Symbol"], y=df["Vega"], mode="lines+markers", name="Vega"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Market closed or outside tracking time (9:20-15:20).")
