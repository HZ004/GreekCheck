import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta, time
from scipy.stats import norm

st.set_page_config(layout="wide")
st.title("Upstox Live Options Greeks Dashboard")

# --- User Input ---
upstox_token = st.text_input("Paste Upstox Access Token", type="password", help="Token expires daily, paste every morning.")

if not upstox_token:
    st.info("Paste your Upstox token to begin.")
    st.stop()

EXCHANGE = "NSE_INDEX"
SYMBOL = "Nifty 50"  # "Nifty 50" or "Bank Nifty" (see Upstox naming for your use case)
STRIKES_TO_PICK = 5  # Each ITM and OTM

API_OP_CONTRACTS = "https://api.upstox.com/v2/option/contract"
API_GREEKS = "https://api.upstox.com/v3/market-quote/option-greek"
API_LTP = "https://api.upstox.com/v3/market-quote/ltp"
HEADERS = {"Authorization": f"Bearer {upstox_token}", "Accept": "application/json"}

# --- Helpers (updated for instrument_key usage) ---
@st.cache_data(ttl=3600, show_spinner="Loading contracts…")
def fetch_option_contracts():
    # Get all contracts (calls and puts, all strikes & expiries) for symbol
    params = {"instrument_key": f"{EXCHANGE}|{SYMBOL}"}
    r = requests.get(API_OP_CONTRACTS, headers=HEADERS, params=params)
    response = r.json()
    if "data" not in response:
        st.error(f"Contracts API failed: {response}")
        st.stop()
    df = pd.DataFrame(response["data"])
    # keep only necessary columns, ensure types
    df["strike_price"] = df["strike_price"].astype(float)
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    return df

@st.cache_data(ttl=30)
def fetch_spot_price(spot_instrument_key):
    r = requests.get(API_LTP, headers=HEADERS, params={"instrument_key": spot_instrument_key})
    response = r.json()
    # Key with colon to match API data keys
    key_in_data = spot_instrument_key.replace("|", ":")
    if "data" in response and key_in_data in response["data"]:
        return float(response["data"][key_in_data].get("last_price", 0))
    st.error(f"Error fetching spot price: {response}")
    st.stop()



def get_nearest_expiry(contract_df):
    today = datetime.now().date()
    return contract_df[contract_df["expiry"] >= today]["expiry"].min()

def select_option_strikes(contract_df, spot, expiry, n=5):
    # Get ordered strikes for calls/puts at given expiry
    df_expiry = contract_df[contract_df["expiry"] == expiry]
    strikes = np.sort(df_expiry["strike_price"].unique())
    idx_atm = (np.abs(strikes - spot)).argmin()
    # ITM < ATM < OTM (for calls; reverse for puts)
    ce_itm = strikes[max(0, idx_atm-n):idx_atm][::-1]  # Calls, In-the-money
    ce_otm = strikes[idx_atm+1:idx_atm+1+n]            # Calls, Out-the-money
    pe_itm = strikes[idx_atm+1:idx_atm+1+n]            # Puts, In-the-money
    pe_otm = strikes[max(0, idx_atm-n):idx_atm][::-1]  # Puts, Out-the-money
    # Gather instrument_keys for each side
    contract = lambda strike, inst_type: df_expiry[(df_expiry["strike_price"] == strike) & (df_expiry["instrument_type"] == inst_type)].iloc[0]

    selection = []
    for strike in ce_itm: selection.append(contract(strike, "CE"))
    for strike in ce_otm: selection.append(contract(strike, "CE"))
    for strike in pe_itm: selection.append(contract(strike, "PE"))
    for strike in pe_otm: selection.append(contract(strike, "PE"))
    return pd.DataFrame(selection)

def poll_greeks_ltp(inst_keys):
    # Bulk poll Greeks and LTPs
    ikeys_str = ",".join(inst_keys)
    data = {}
    # query LTP
    r = requests.get(API_LTP, headers=HEADERS, params={"instrument_key": ikeys_str})
    ltp_resp = r.json().get("data", {})
    # query Greeks
    r = requests.get(API_GREEKS, headers=HEADERS, params={"instrument_key": ikeys_str})
    greeks_resp = r.json().get("data", {})
    for ikey in inst_keys:
        ltp = ltp_resp.get(ikey, {}).get("ltp", np.nan)
        g   = greeks_resp.get(ikey, {})
        data[ikey] = {"ltp": ltp}
        for greek in ["delta", "gamma", "vega", "theta", "iv"]:
            data[ikey][greek] = g.get(greek, None)
    return data

# Black-Scholes fallback (uses spot, strike, t, r, iv)
def black_scholes_greeks(S, K, T, sigma, option_type, r=0.05):
    if T <= 0 or sigma <= 0:  # avoid math errors
        return dict(delta=0, gamma=0, vega=0, theta=0)
    d1 = (np.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = option_type == "CE"
    delta = norm.cdf(d1) if call else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    theta_call = (-S * norm.pdf(d1) * sigma/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2))/365
    theta_put  = (-S * norm.pdf(d1) * sigma/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2))/365
    theta = theta_call if call else theta_put
    return dict(delta=delta, gamma=gamma, vega=vega, theta=theta)

def get_years_to_expiry(expiry_date):
    now = datetime.now()
    expiry = datetime.combine(expiry_date, time(15,30))
    return max(1e-6, (expiry-now).total_seconds() / (365.0 * 86400))

def fallback_compute(contract, spot, ltp):
    # Fallback Black-Scholes if greeks missing
    K = contract["strike_price"]
    T = get_years_to_expiry(contract["expiry"])
    option_type = contract["option_type"]
    sigma = 0.2  # Assumed IV for fallback; can try guess from ltp
    return black_scholes_greeks(spot, K, T, sigma, option_type)

# ----- Session State/buffering -----
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

now = datetime.now(IST)
today = now.date()
strike_lock_time = datetime.combine(today, time(9, 16), IST)

st.sidebar.info("Strikes are fixed at 09:16 each day.")

start_poll = datetime.combine(today, time(9,20), IST)
end_poll = datetime.combine(today, time(15,20), IST)

contract_df = fetch_option_contracts()

# Use underlying instrument_key ("NSE_INDEX|Nifty 50") for spot price
spot_price = fetch_spot_price(f"{EXCHANGE}|{SYMBOL}")
expiry_list = sorted(contract_df["expiry"].unique())

expiry = st.selectbox("Option Expiry", expiry_list, index=expiry_list.index(get_nearest_expiry(contract_df)))
show_spot = st.markdown(f"**Spot Price:** {spot_price}")

# Fix strikes at 09:16 IST, update once per day
if "strike_df" not in st.session_state or st.session_state.get("strikes_for_day") != (str(today), expiry):
    if now < strike_lock_time:
        st.info("Waiting for strike selection at 09:16 IST…")
        st.stop()
    else:
        # After 09:16, select strikes immediately on startup, using current spot
        sel_df = select_option_strikes(contract_df, spot_price, expiry, n=STRIKES_TO_PICK)
        st.session_state["strike_df"] = sel_df.copy()
        st.session_state["strikes_for_day"] = (str(today), expiry)


display_df = st.session_state["strike_df"]
st.table(display_df[["option_type", "strike_price", "expiry", "instrument_key"]].sort_values("option_type"))
keys_monitored = list(display_df.instrument_key)

# --- Live Data Polling ---
if start_poll <= now <= end_poll:
    placeh = st.empty()
    datalist = st.session_state.get("greek_ts", [])

    # Poll APIs (Greeks and LTP)
    greek_data = poll_greeks_ltp(keys_monitored)
    timestamp = datetime.now()
    # Assemble time-series row
    row = {"timestamp": timestamp}
    for i, contract in display_df.iterrows():
        ikey = contract["instrument_key"]
        # Use API values, fallback to Black-Scholes if needed
        gd = greek_data.get(ikey, {})
        ltp = gd.get("ltp", np.nan)
        row.update({f"{contract['option_type']}_{int(contract['strike_price'])}_{k}": (
            gd[k] if gd.get(k) not in [None, ""] else
            fallback_compute(contract, spot_price, ltp).get(k, np.nan))
            for k in ["delta","gamma","vega","theta","iv"]})
    datalist.append(row)
    st.session_state["greek_ts"] = datalist
    # Display DataFrame and charts
    df = pd.DataFrame(datalist)
    st.dataframe(df.tail(50))
    for metric in ["delta","gamma","vega","theta","iv"]:
        chosen = [c for c in df.columns if c.endswith(f"_{metric}")]
        fig = px.line(df, x="timestamp", y=chosen, title=f"{metric.upper()} Time Series")
        placeh.plotly_chart(fig, use_container_width=True)
    st.experimental_rerun()
else:
    st.info("Live polling active only between 09:20 and 15:20 IST.")

st.caption("Uses Upstox official APIs. Token expires daily.")

# ---------- End of file ----------
