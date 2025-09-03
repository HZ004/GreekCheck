import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta, time
from scipy.stats import norm

# --- User Inputs ---
st.title("Upstox Options Greeks Dashboard")
upstox_token = st.text_input("Paste Upstox Access Token", type="password")

# Constants
EXCHANGE = "NSE_INDEX"
SYMBOL = "NIFTY"  # Change to BANKNIFTY if needed
STRIKES_TO_PICK = 5  # Each ITM and OTM
API_BASE = "https://api.upstox.com"
option_contracts_url = f"{API_BASE}/v3/market/option-contracts"
option_greeks_url = f"{API_BASE}/v3/market-quote/option-greek"
ltp_url = f"{API_BASE}/v3/market-quote/ltp"
HEADERS = {"Authorization": f"Bearer {upstox_token}"}

# --- Helper Functions ---
def get_market_expiries_and_strikes():
    '''Fetch option contracts (strikes + expiries) from Upstox'''
    resp = requests.get(option_contracts_url, headers=HEADERS, params={"exchange": EXCHANGE, "symbol": SYMBOL})
    st.write("Raw API response:", resp.text)
    contracts = resp.json()["contracts"]
    # Filter by expiry: selecting the nearest by default
    expiry_dates = sorted(set(c["expiry"] for c in contracts))
    # UI for override expiry
    expiry = st.selectbox("Expiry date", expiry_dates)
    strikes = sorted(set(c["strike"] for c in contracts if c["expiry"] == expiry))
    return strikes, expiry

def get_spot_price():
    '''Fetch spot price (ATM reference) from Upstox'''
    params = {"exchange": EXCHANGE, "symbol": SYMBOL}
    resp = requests.get(ltp_url, headers=HEADERS, params=params)
    data = resp.json()
    return float(data["ltp"])

def select_strikes(strikes, atm, n=STRIKES_TO_PICK):
    '''Choose ITM/OTM strikes for CE & PE'''
    strikes = np.array(strikes)
    idx_atm = (np.abs(strikes - atm)).argmin()
    ce_itm = strikes[max(0, idx_atm-n):idx_atm][::-1]
    ce_otm = strikes[idx_atm+1:idx_atm+1+n]
    pe_itm = strikes[idx_atm+1:idx_atm+1+n]
    pe_otm = strikes[max(0, idx_atm-n):idx_atm][::-1]
    return ce_itm, ce_otm, pe_itm, pe_otm

def get_greeks(strike, side, expiry):
    '''Fetch Greeks from Upstox or compute via Black-Scholes'''
    instrument_type = "CE" if side == "call" else "PE"
    params = {
        "exchange": EXCHANGE,
        "symbol": SYMBOL,
        "expiry": expiry,
        "strike": strike,
        "instrument_type": instrument_type,
    }
    try:
        resp = requests.get(option_greeks_url, headers=HEADERS, params=params, timeout=3)
        data = resp.json()
        if all(k in data for k in ["delta", "gamma", "vega", "theta", "iv"]):
            return {k: data[k] for k in ["delta", "gamma", "vega", "theta", "iv"]}
    except Exception:
        pass
    # Fallback: get LTP and compute Greeks locally
    option_price = get_option_price(strike, side, expiry)
    spot = get_spot_price()
    time_to_expiry = get_years_to_expiry(expiry)
    iv = estimate_iv(option_price, spot, strike, time_to_expiry, side)
    greeks = black_scholes_greeks(spot, strike, time_to_expiry, iv, side)
    greeks["iv"] = iv
    return greeks

def get_option_price(strike, side, expiry):
    '''Fetch LTP for option'''
    instrument_type = "CE" if side == "call" else "PE"
    params = {
        "exchange": EXCHANGE,
        "symbol": SYMBOL,
        "expiry": expiry,
        "strike": strike,
        "instrument_type": instrument_type,
    }
    resp = requests.get(ltp_url, headers=HEADERS, params=params)
    data = resp.json()
    return float(data["ltp"])

def get_years_to_expiry(expiry):
    '''Convert expiry to year fraction'''
    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
    now = datetime.now()
    return max(1e-6, (expiry_dt - now).days / 365.0)

def estimate_iv(option_price, spot, strike, t, side):
    '''Naive estimation via Black-Scholes implied volatility inversion'''
    # For simplicity, fixed r = 0.05
    from scipy.optimize import root_scalar
    def bs_price(iv):
        return black_scholes_price(spot, strike, t, 0.05, iv, side)
    try:
        result = root_scalar(lambda iv: bs_price(iv) - option_price, bracket=[0.01, 1.0])
        return float(result.root) if result.converged else 0.2
    except Exception:
        return 0.2

def black_scholes_price(S, K, T, r, sigma, side):
    '''Black-Scholes price formula'''
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if side == "call":
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def black_scholes_greeks(S, K, T, sigma, side):
    '''Greeks: delta, gamma, vega, theta'''
    r = 0.05
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1) if side == "call" else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    theta_call = (-S * norm.pdf(d1) * sigma/(2*np.sqrt(T)) - r*K*np.exp(-r*T) * norm.cdf(d2)) / 365
    theta_put = (-S * norm.pdf(d1) * sigma/(2*np.sqrt(T)) + r*K*np.exp(-r*T) * norm.cdf(-d2)) / 365
    theta = theta_call if side == "call" else theta_put
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


# --- Polling & Data Storage ---
if not upstox_token:
    st.info("Paste your Upstox access token to begin")
    st.stop()

today = datetime.now().date()
start_poll = datetime.combine(today, time(9,20))
end_poll = datetime.combine(today, time(15,20))
strikes, expiry = get_market_expiries_and_strikes()
spot = get_spot_price()
ce_itm, ce_otm, pe_itm, pe_otm = select_strikes(strikes, spot)

# Store option keys and dataframes in session state
if "strike_set" not in st.session_state:
    if datetime.now().time() >= time(9,16):
        st.session_state.strike_set = {
            "CE_ITM": list(ce_itm),
            "CE_OTM": list(ce_otm),
            "PE_ITM": list(pe_itm),
            "PE_OTM": list(pe_otm)
        }
    else:
        st.info("Strikes will be selected/fixed at 09:16 IST each day.")

strike_dict = st.session_state.get("strike_set", {})
polling_active = start_poll.time() <= datetime.now().time() <= end_poll.time()

poll_interval = 1  # seconds
poll_time = st.empty()
data = st.session_state.get("greek_data", pd.DataFrame())

if polling_active:
    poll_time.info(f"Polling Upstox an option-greeks every {poll_interval}s")
    rows = []
    for side in ["call", "put"]:
        for itm_key, otm_key in [("CE_ITM", "CE_OTM"), ("PE_ITM", "PE_OTM")]:
            strikes_selected = strike_dict.get(itm_key, []) + strike_dict.get(otm_key, [])
            for strike in strikes_selected:
                greeks = get_greeks(strike, side, expiry)
                entry = {
                    "Timestamp": datetime.now(),
                    "Strike": strike,
                    "Type": "CE" if side == "call" else "PE",
                    **greeks
                }
                rows.append(entry)
    if len(rows) > 0:
        df = pd.DataFrame(rows)
        if data is not None and not data.empty:
            data = pd.concat([data, df], ignore_index=True)
        else:
            data = df
        st.session_state["greek_data"] = data

# --- UI: Display Tables/Charts ---
if data is not None and not data.empty:
    st.dataframe(data.tail(50))
    # Plot Greeks over time
    for greek in ["delta", "gamma", "vega", "theta", "iv"]:
        fig = px.line(
            data,
            x="Timestamp",
            y=greek,
            color="Strike",
            title=f"{greek.capitalize()} over time"
        )
        st.plotly_chart(fig, use_container_width=True)
    st.experimental_rerun()  # refresh every second

else:
    st.info("Waiting for option Greek data. Market polling is live from 09:20 to 15:20 IST.")

# --- End of file ---
