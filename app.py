import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, time as dtime
from upstox_api.api import Upstox
import pytz
import time
import math

# ==========================
# Black-Scholes Greeks
# ==========================
def black_scholes_greeks(option_type, S, K, T, r, sigma):
    """Calculate Delta, Gamma, Theta, Vega using Black-Scholes."""
    from scipy.stats import norm

    if T <= 0 or sigma <= 0:
        return 0, 0, 0, 0

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == 'CE':
        delta = norm.cdf(d1)
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
                 - r * K * math.exp(-r * T) * norm.cdf(d2))
    else:  # PE
        delta = -norm.cdf(-d1)
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
                 + r * K * math.exp(-r * T) * norm.cdf(-d2))

    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * norm.pdf(d1) * math.sqrt(T)

    return delta, gamma, theta, vega

# ==========================
# Upstox Connection
# ==========================
@st.cache_resource
def connect_upstox(access_token):
    u = Upstox('6d211fe5-cf98-45b8-b1fd-fcd4b3865793', 'ab7tsjvh2s')  # Replace with your keys
    u.set_access_token(access_token)
    return u

# ==========================
# Utility Functions
# ==========================
def get_nifty_spot(u):
    ltp = u.get_live_feed('NSE_INDEX|Nifty 50', 'LTP')
    return ltp['ltp']

def get_expiries(u, symbol="NIFTY"):
    inst = u.get_master_contract('NSE_FO')
    expiry_list = sorted(list({i.expiry for i in inst if i.symbol == symbol}))
    return expiry_list[:4]  # Next 4 expiries

def get_option_contracts(u, spot, expiry, symbol="NIFTY"):
    inst = u.get_master_contract('NSE_FO')
    strikes = sorted({i.strike_price for i in inst if i.symbol == symbol and i.expiry == expiry})
    atm_strike = min(strikes, key=lambda x: abs(x - spot))

    # Choose 5 ITM + 5 OTM for CE and PE separately
    all_strikes = sorted(strikes)
    idx = all_strikes.index(atm_strike)
    selected_strikes = all_strikes[max(idx-5,0):idx+6]  # +/-5 strikes

    contracts = []
    for s in selected_strikes:
        for opt_type in ['CE', 'PE']:
            token = next((i.token for i in inst if i.symbol == symbol and i.expiry == expiry and i.strike_price == s and i.instrument_type == opt_type), None)
            if token:
                contracts.append({
                    'symbol': symbol,
                    'strike': s,
                    'type': opt_type,
                    'token': token
                })
    return contracts

# ==========================
# Streamlit UI
# ==========================
st.set_page_config(layout="wide")
st.title("📈 NIFTY Options Live Greeks Dashboard")

token = st.text_input("🔑 Enter your Upstox Access Token", type="password")
if not token:
    st.warning("Please enter your token to start.")
    st.stop()

u = connect_upstox(token)

expiries = get_expiries(u)
expiry_choice = st.selectbox("📅 Choose Expiry", expiries, index=0)

# Fetch initial contracts at 9:16 AM daily
now_ist = datetime.now(pytz.timezone("Asia/Kolkata"))
if now_ist.time() >= dtime(9, 16):
    nifty_spot = get_nifty_spot(u)
    contracts = get_option_contracts(u, nifty_spot, expiry_choice)
else:
    st.info("Contracts will refresh at 09:16 AM IST.")
    st.stop()

# ==========================
# Live Update Loop
# ==========================
placeholder = st.empty()

def within_market_hours():
    t = datetime.now(pytz.timezone("Asia/Kolkata")).time()
    return dtime(9, 20) <= t <= dtime(15, 20)

st.info("Live updates start at 09:20 AM and stop at 03:20 PM IST.")

while True:
    if not within_market_hours():
        time.sleep(5)
        continue

    data = []
    for c in contracts:
        ltp_data = u.get_live_feed(c['token'], 'LTP')
        ltp = ltp_data['ltp']
        iv = ltp_data.get('implied_volatility', 0.2)  # fallback if API doesn't give IV
        expiry_days = (expiry_choice - now_ist.date()).days
        T = max(expiry_days / 365, 1/365)
        delta, gamma, theta, vega = black_scholes_greeks(c['type'], nifty_spot, c['strike'], T, 0.05, iv)
        data.append({
            'strike': c['strike'],
            'type': c['type'],
            'ltp': ltp,
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega
        })

    df = pd.DataFrame(data)

    fig = go.Figure()
    for greek in ['delta', 'gamma', 'theta', 'vega']:
        fig.add_trace(go.Scatter(
            x=[f"{row['strike']}{row['type']}" for _, row in df.iterrows()],
            y=df[greek],
            mode='lines+markers',
            name=greek
        ))

    fig.update_layout(title="Live Greeks", xaxis_title="Option", yaxis_title="Value", legend_title="Greek")

    with placeholder.container():
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df)

    time.sleep(1)