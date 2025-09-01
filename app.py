import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.express as px
import websocket
import threading
import time
from datetime import datetime, timedelta
from scipy.stats import norm

# ===============================
# CONFIG
# ===============================
INDEX_SYMBOL = "NIFTY50"
LOTSIZE = 50
MARKET_START = "09:20"
MARKET_END = "15:20"

# ===============================
# BLACK-SCHOLES FORMULAS
# ===============================
def black_scholes_greeks(S, K, T, r, sigma, option_type):
    if T <= 0 or sigma <= 0:
        return {"delta": 0, "gamma": 0, "vega": 0, "theta": 0, "rho": 0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "CE":
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        delta = -norm.cdf(-d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}

# ===============================
# HELPER: GET OPTION CHAIN
# ===============================
def get_instruments(access_token):
    url = "https://api-v2.upstox.com/instruments"
    headers = {"Authorization": f"Bearer {access_token}"}
    r = requests.get(url, headers=headers)
    data = r.json()
    df = pd.DataFrame(data)
    return df

def get_nearest_expiry(df):
    expiries = sorted(df['expiry'].unique())
    return expiries[0], expiries[:4]

def pick_strikes(df, spot):
    strikes = sorted(df['strike'].unique())
    closest = min(strikes, key=lambda x: abs(x - spot))
    idx = strikes.index(closest)
    return strikes[max(0, idx-5):idx+6]  # 5 OTM, 5 ITM

# ===============================
# WEBSOCKET CLIENT
# ===============================
class UpstoxWS:
    def __init__(self, token, instruments):
        self.token = token
        self.instruments = instruments
        self.data = {}
        self.ws = None

    def on_message(self, ws, message):
        msg = json.loads(message)
        for tick in msg.get("data", []):
            instrument_key = tick["instrument_key"]
            ltp = tick["last_price"]
            self.data[instrument_key] = ltp

    def on_open(self, ws):
        sub = {
            "guid": "abc123",
            "method": "sub",
            "data": {"instrumentKeys": self.instruments}
        }
        ws.send(json.dumps(sub))

    def start(self):
        url = "wss://api-v2.upstox.com/feed/market-data-feed"
        headers = [f"Authorization: Bearer {self.token}"]
        self.ws = websocket.WebSocketApp(
            url, header=headers,
            on_message=self.on_message, on_open=self.on_open
        )
        threading.Thread(target=self.ws.run_forever, daemon=True).start()

# ===============================
# STREAMLIT APP
# ===============================
st.title("📈 Live NIFTY50 Option Greeks")

access_token = st.text_input("Enter Upstox Access Token", type="password")

if access_token:
    df = get_instruments(access_token)
    spot_price = df[(df['symbol'] == INDEX_SYMBOL) & (df['instrument_type'] == 'INDEX')]['tick_size'].mean() or 20000
    nearest_expiry, expiries = get_nearest_expiry(df)

    expiry_choice = st.selectbox("Choose Expiry", expiries, index=0)

    strikes_today = pick_strikes(df[df['expiry'] == expiry_choice], spot_price)
    selected = df[(df['expiry'] == expiry_choice) & 
                  (df['strike'].isin(strikes_today)) & 
                  (df['instrument_type'].isin(['OPTIDX']))]

    instrument_keys = selected['instrument_key'].tolist()
    ws_client = UpstoxWS(access_token, instrument_keys)
    ws_client.start()

    placeholder = st.empty()
    greek_history = []

    while True:
        now = datetime.now().strftime("%H:%M")
        if now < MARKET_START or now > MARKET_END:
            time.sleep(1)
            continue

        live_prices = {k: ws_client.data.get(k, np.nan) for k in instrument_keys}
        selected['ltp'] = selected['instrument_key'].map(live_prices)

        # Greeks calculation
        rows = []
        for _, row in selected.iterrows():
            greeks = black_scholes_greeks(
                S=spot_price,
                K=row['strike'],
                T=(datetime.strptime(row['expiry'], "%Y-%m-%d") - datetime.now()).days / 365,
                r=0.05,
                sigma=0.2,
                option_type=row['option_type']
            )
            rows.append({**row, **greeks})

        greeks_df = pd.DataFrame(rows)
        greek_history.append(greeks_df[['strike', 'option_type', 'delta', 'gamma', 'vega', 'theta', 'rho']])

        # Plot Delta
        fig = px.line(greeks_df, x="strike", y="delta", color="option_type", title="Delta Trend")
        placeholder.plotly_chart(fig, use_container_width=True)
        time.sleep(1)
