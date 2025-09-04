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
SYMBOL = "Nifty 50"
STRIKES_TO_PICK = 5

API_OP_CONTRACTS = "https://api.upstox.com/v2/option/contract"
API_GREEKS = "https://api.upstox.com/v3/market-quote/option-greek"
API_LTP = "https://api.upstox.com/v3/market-quote/ltp"
HEADERS = {"Authorization": f"Bearer {upstox_token}", "Accept": "application/json"}

@st.cache_data(ttl=3600, show_spinner="Loading contracts…")
def fetch_option_contracts():
    params = {"instrument_key": f"{EXCHANGE}|{SYMBOL}"}
    r = requests.get(API_OP_CONTRACTS, headers=HEADERS, params=params)
    response = r.json()
    if "data" not in response:
        st.error(f"Contracts API failed: {response}")
        st.stop()
    df = pd.DataFrame(response["data"])
    df["strike_price"] = df["strike_price"].astype(float)
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    return df

@st.cache_data(ttl=30)
def fetch_spot_price(spot_instrument_key):
    r = requests.get(API_LTP, headers=HEADERS, params={"instrument_key": spot_instrument_key})
    response = r.json()
    key_in_data = spot_instrument_key.replace("|", ":")
    if "data" in response and key_in_data in response["data"]:
        return float(response["data"][key_in_data].get("last_price", 0))
    st.error(f"Error fetching spot price: {response}")
    st.stop()

def get_nearest_expiry(contract_df):
    today = datetime.now().date()
    return contract_df[contract_df["expiry"] >= today]["expiry"].min()

def select_option_strikes(contract_df, spot, expiry, n=5):
    df_expiry = contract_df[contract_df["expiry"] == expiry]
    strikes = np.sort(df_expiry["strike_price"].unique())
    idx_atm = (np.abs(strikes - spot)).argmin()
    atm_strike = strikes[idx_atm]

    ce_itm = strikes[max(0, idx_atm - n):idx_atm][::-1]
    ce_atm = np.array([atm_strike])
    ce_otm = strikes[idx_atm + 1:idx_atm + 1 + n]

    pe_itm = strikes[idx_atm + 1:idx_atm + 1 + n]
    pe_atm = np.array([atm_strike])
    pe_otm = strikes[max(0, idx_atm - n):idx_atm][::-1]

    ce_strikes = np.concatenate([ce_itm, ce_atm, ce_otm])
    pe_strikes = np.concatenate([pe_itm, pe_atm, pe_otm])

    contract = lambda strike, inst_type: df_expiry[
        (df_expiry["strike_price"] == strike) & (df_expiry["instrument_type"] == inst_type)
    ].iloc[0]

    selection = []
    for strike in ce_strikes:
        selection.append(contract(strike, "CE"))
    for strike in pe_strikes:
        selection.append(contract(strike, "PE"))

    return pd.DataFrame(selection)

def poll_greeks_ltp(inst_keys):
    ikeys_str = ",".join(inst_keys)
    data = {}
    r = requests.get(API_LTP, headers=HEADERS, params={"instrument_key": ikeys_str})
    ltp_resp = r.json().get("data", {})
    r = requests.get(API_GREEKS, headers=HEADERS, params={"instrument_key": ikeys_str})
    greeks_resp = r.json().get("data", {})
    for ikey in inst_keys:
        ltp = ltp_resp.get(ikey, {}).get("ltp", np.nan)
        g = greeks_resp.get(ikey, {})
        data[ikey] = {"ltp": ltp}
        for greek in ["delta", "gamma", "vega", "theta", "rho"]:
            data[ikey][greek] = g.get(greek, None)
    return data

def black_scholes_greeks(S, K, T, sigma, instrument_type, r=0.05):
    if T <= 0 or sigma <= 0:
        return dict(delta=0, gamma=0, vega=0, theta=0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = instrument_type == "CE"
    delta = norm.cdf(d1) if call else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    theta_call = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    theta_put = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    theta = theta_call if call else theta_put
    return dict(delta=delta, gamma=gamma, vega=vega, theta=theta)

def get_years_to_expiry(expiry_date):
    now = datetime.now()
    expiry = datetime.combine(expiry_date, time(15, 30))
    return max(1e-6, (expiry - now).total_seconds() / (365.0 * 86400))

def fallback_compute(contract, spot, ltp):
    K = contract["strike_price"]
    T = get_years_to_expiry(contract["expiry"])
    instrument_type = contract["instrument_type"]
    sigma = 0.2
    return black_scholes_greeks(spot, K, T, sigma, instrument_type)

IST = ZoneInfo("Asia/Kolkata")
now = datetime.now(IST)
today = now.date()
strike_lock_time = datetime.combine(today, time(9, 16), IST)
start_poll = datetime.combine(today, time(9, 20), IST)
end_poll = datetime.combine(today, time(15, 20), IST)

contract_df = fetch_option_contracts()
spot_price = fetch_spot_price(f"{EXCHANGE}|{SYMBOL}")
expiry_list = sorted(contract_df["expiry"].unique())
expiry = st.selectbox("Option Expiry", expiry_list, index=expiry_list.index(get_nearest_expiry(contract_df)))

st.sidebar.info("Strikes are fixed at 09:16 each day.")

if "strike_df" not in st.session_state or st.session_state.get("strikes_for_day") != (str(today), expiry):
    if now < strike_lock_time:
        st.info("Waiting for strike selection at 09:16 IST…")
        st.stop()
    else:
        sel_df = select_option_strikes(contract_df, spot_price, expiry, n=STRIKES_TO_PICK)
        filtered_df = sel_df[sel_df["strike_price"] % 100 == 0].sort_values(by=["instrument_type", "strike_price"])
        st.session_state["strike_df"] = filtered_df.copy()
        st.session_state["strikes_for_day"] = (str(today), expiry)

display_df = st.session_state["strike_df"]
keys_monitored = list(display_df["instrument_key"])

refresh_count = st_autorefresh(interval=5000, limit=1000, key="greeks_refresh")

if start_poll <= now <= end_poll:
    datalist = st.session_state.get("greek_ts", [])

    greek_data = poll_greeks_ltp(keys_monitored)
    timestamp = datetime.now(IST)

    row = {"timestamp": timestamp}
    for _, contract in display_df.iterrows():
        ikey = contract["instrument_key"]
        gd = greek_data.get(ikey, {})
        ltp = gd.get("ltp", float("nan"))
        row.update(
            {
                f"{contract['instrument_type']}_{int(contract['strike_price'])}_{k}": (gd.get(k) if gd.get(k) not in [None, ""] else fallback_compute(contract, spot_price, ltp).get(k, float("nan")))
                for k in ["delta", "gamma", "vega", "theta", "rho"]
            }
        )

    datalist.append(row)
    st.session_state["greek_ts"] = datalist

    df = pd.DataFrame(datalist)

    styled_df = display_df[["instrument_type", "strike_price", "expiry"]].copy()
    styled_df["strike_price"] = styled_df["strike_price"].astype(int)
    styled_df["expiry"] = pd.to_datetime(styled_df["expiry"]).dt.strftime("%Y-%m-%d")
    styled_df = styled_df.rename(columns={"instrument_type": "Option Type", "strike_price": "Strike Price", "expiry": "Expiry Date"})

    def highlight_option_type(row):
        if row["Option Type"] == "CE":
            return ["background-color: #d0e7ff"] * len(row)
        elif row["Option Type"] == "PE":
            return ["background-color: #ffd6d6"] * len(row)
        else:
            return [""] * len(row)

    st.dataframe(styled_df.style.apply(highlight_option_type, axis=1), height=400, use_container_width=True)
	
	# Your DataFrame 'df' from polling time series contains columns like 'CE_18500_delta', 'PE_18500_delta', etc.
	
	greek_metrics = ["delta", "theta", "vega", "rho"]  # Including rho
	option_types = ["CE", "PE"]
	names_for_caption = {
	    "delta": "Delta",
	    "theta": "Theta",
	    "vega": "Vega",
	    "rho": "Rho"
	}
	
	# Create 2 columns for side-by-side CE and PE charts
	col1, col2 = st.columns(2)
	
	for row_idx, metric in enumerate(greek_metrics):
	    # Find relevant CE columns and PE columns for the metric
	    ce_cols = [col for col in df.columns if (col.startswith("CE_") and col.endswith(f"_{metric}"))]
	    pe_cols = [col for col in df.columns if (col.startswith("PE_") and col.endswith(f"_{metric}"))]
	
	    # Plot CE metric in left column
	    with col1:
	        if ce_cols:
	            fig_ce = px.line(
	                df,
	                x="timestamp",
	                y=ce_cols,
	                title=f"Call (CE) {names_for_caption[metric]} Time Series",
	                labels={"value": names_for_caption[metric], "timestamp": "Time"},
	            )
	            st.plotly_chart(fig_ce, use_container_width=True)
	
	    # Plot PE metric in right column
	    with col2:
	        if pe_cols:
	            fig_pe = px.line(
	                df,
	                x="timestamp",
	                y=pe_cols,
	                title=f"Put (PE) {names_for_caption[metric]} Time Series",
	                labels={"value": names_for_caption[metric], "timestamp": "Time"},
	            )
            st.plotly_chart(fig_pe, use_container_width=True)


    if not df.empty:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode()

        st.download_button(
            label="Download Full Day Greeks CSV",
            data=csv_bytes,
            file_name=f"greeks_data_{now.strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
else:
    st.info("Live polling active only between 09:20 and 15:20 IST.")
