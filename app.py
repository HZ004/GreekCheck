import time as pytime
from datetime import datetime, timedelta, time
from streamlit_autorefresh import st_autorefresh
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import norm
from zoneinfo import ZoneInfo
import io


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

# def select_option_strikes(contract_df, spot, expiry, n=5):
#     # Get ordered strikes for calls/puts at given expiry
#     df_expiry = contract_df[contract_df["expiry"] == expiry]
#     strikes = np.sort(df_expiry["strike_price"].unique())
#     idx_atm = (np.abs(strikes - spot)).argmin()
#     # ITM < ATM < OTM (for calls; reverse for puts)
#     ce_itm = strikes[max(0, idx_atm-n):idx_atm][::-1]  # Calls, In-the-money
#     ce_otm = strikes[idx_atm+1:idx_atm+1+n]            # Calls, Out-the-money
#     pe_itm = strikes[idx_atm+1:idx_atm+1+n]            # Puts, In-the-money
#     pe_otm = strikes[max(0, idx_atm-n):idx_atm][::-1]  # Puts, Out-the-money
#     # Gather instrument_keys for each side
#     contract = lambda strike, inst_type: df_expiry[(df_expiry["strike_price"] == strike) & (df_expiry["instrument_type"] == inst_type)].iloc[0]

#     selection = []
#     for strike in ce_itm: selection.append(contract(strike, "CE"))
#     for strike in ce_otm: selection.append(contract(strike, "CE"))
#     for strike in pe_itm: selection.append(contract(strike, "PE"))
#     for strike in pe_otm: selection.append(contract(strike, "PE"))
#     return pd.DataFrame(selection)

def select_option_strikes(contract_df, spot, expiry, n=5):
    # Filter contracts for expiry
    df_expiry = contract_df[contract_df["expiry"] == expiry]
    strikes = np.sort(df_expiry["strike_price"].unique())
    idx_atm = (np.abs(strikes - spot)).argmin()
    atm_strike = strikes[idx_atm]

    # ITM < ATM < OTM (for calls; reverse for puts)
    ce_itm = strikes[max(0, idx_atm - n):idx_atm][::-1]  # Calls, In-the-money
    ce_atm = np.array([atm_strike])                        # Calls, At-the-money
    ce_otm = strikes[idx_atm + 1:idx_atm + 1 + n]         # Calls, Out-of-the-money

    pe_itm = strikes[idx_atm + 1:idx_atm + 1 + n]         # Puts, In-the-money
    pe_atm = np.array([atm_strike])                        # Puts, At-the-money
    pe_otm = strikes[max(0, idx_atm - n):idx_atm][::-1]   # Puts, Out-of-the-money

    # Combine strikes for calls and puts, including ATM strikes
    ce_strikes = np.concatenate([ce_itm, ce_atm, ce_otm])
    pe_strikes = np.concatenate([pe_itm, pe_atm, pe_otm])

    # Lambda function to get contract row by strike and option type
    contract = lambda strike, inst_type: df_expiry[
        (df_expiry["strike_price"] == strike) & (df_expiry["instrument_type"] == inst_type)
    ].iloc[0]

    # Build selection list
    selection = []
    for strike in ce_strikes:
        selection.append(contract(strike, "CE"))
    for strike in pe_strikes:
        selection.append(contract(strike, "PE"))

    # Return combined DataFrame of selection
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
def black_scholes_greeks(S, K, T, sigma, instrument_type, r=0.05):
    if T <= 0 or sigma <= 0:  # avoid math errors
        return dict(delta=0, gamma=0, vega=0, theta=0)
    d1 = (np.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = instrument_type == "CE"
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
    instrument_type = contract["instrument_type"]
    sigma = 0.2  # Assumed IV for fallback; can try guess from ltp
    return black_scholes_greeks(spot, K, T, sigma, instrument_type)

# ----- Session State/buffering -----

# Set up IST timezone
IST = ZoneInfo("Asia/Kolkata")

# Auto-refresh every 5 seconds, max 1000 refreshes
refresh_count = st_autorefresh(interval=5000, limit=1000, key="greeks_refresh")

now = datetime.now(IST)
today = now.date()

start_poll = datetime.combine(today, time(9, 20), IST)
end_poll = datetime.combine(today, time(15, 20), IST)

table_placeholder = st.empty()

if start_poll <= now <= end_poll:
    # Retrieve or initialize Greeks time series list
    datalist = st.session_state.get("greek_ts", [])

    # Poll Greeks once per run
    greek_data = poll_greeks_ltp(keys_monitored)
    timestamp = datetime.now(IST)

    # Build new row for current timestamp
    row = {"timestamp": timestamp}
    for i, contract in display_df.iterrows():
        ikey = contract["instrument_key"]
        gd = greek_data.get(ikey, {})
        ltp = gd.get("ltp", float('nan'))
        row.update({
            f"{contract['instrument_type']}_{int(contract['strike_price'])}_{k}":
            (gd.get(k) if gd.get(k) not in [None, ""] else fallback_compute(contract, spot_price, ltp).get(k, float('nan')))
            for k in ["delta", "gamma", "vega", "theta", "rho"]
        })

    datalist.append(row)
    st.session_state["greek_ts"] = datalist

    df = pd.DataFrame(datalist)

    # Styled table of strikes
    styled_df = display_df[["instrument_type", "strike_price", "expiry"]].copy()

    filtered_df = display_df[display_df["strike_price"] % 100 == 0].sort_values(by=['instrument_type', 'strike_price'])
    display_df = filtered_df  # if you want further usage

    keys_monitored = list(display_df["instrument_key"])

    # Plot Greeks (delta, theta, vega, rho) in 2x2 layout
    greek_metrics = ["delta", "theta", "vega", "rho"]
    names_for_caption = {"delta": "Delta", "theta": "Theta", "vega": "Vega", "rho": "Rho"}
    col1, col2 = st.columns(2)
    metric_cols = [col1, col2, col1, col2]

    for idx, metric in enumerate(greek_metrics):
        with metric_cols[idx]:
            chosen = [c for c in df.columns if c.endswith(f"_{metric}")]
            if chosen:
                fig = px.line(
                    df,
                    x="timestamp",
                    y=chosen,
                    title=f"{names_for_caption[metric]} Time Series",
                    labels={"value": names_for_caption[metric], "timestamp": "Time"},
                )
                st.plotly_chart(fig, use_container_width=True)

    # Download CSV button for full Greeks timeseries
    if not df.empty:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode()

        st.download_button(
            label="Download Full Day Greeks CSV",
            data=csv_bytes,
            file_name=f"greeks_data_{now.strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

else:
    st.info("Live polling active only between 09:20 and 15:20 IST.")
