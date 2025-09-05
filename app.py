import time as pytime
from datetime import datetime, timedelta, time
from streamlit_autorefresh import st_autorefresh
import streamlit as st
import requests
import pandas as pd
import numpy as np
from scipy.stats import norm
from zoneinfo import ZoneInfo
import io
import os
import json

st.set_page_config(layout="wide")
st.title("Upstox Live Options Greeks Dashboard")

# --- User Input ---
upstox_token = os.getenv("UPSTOX_TOKEN")
if not upstox_token:
    st.error("Upstox access token not found in environment variables. Please set UPSTOX_TOKEN in Render environment.")
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
    ltp_map = {v.get("instrument_token"): v.get("last_price", np.nan) for v in ltp_resp.values() if v.get("instrument_token")}
    greeks_map = {v.get("instrument_token"): v for v in greeks_resp.values() if v.get("instrument_token")}
    for ikey in inst_keys:
        ltp = ltp_map.get(ikey, np.nan)
        g = greeks_map.get(ikey, {})
        data[ikey] = {"ltp": ltp}
        for greek in ["delta", "gamma", "theta"]:
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
strike_lock_time = datetime.combine(today, time(9, 20), IST)
start_poll = datetime.combine(today, time(9, 20), IST)
end_poll = datetime.combine(today, time(15, 20), IST)

spot_instrument_key = f"{EXCHANGE}|{SYMBOL}"
spot_price = fetch_spot_price(spot_instrument_key)
st.markdown(f"""
### Current {SYMBOL} LTP: <span style='color:green; font-weight:bold;'>{spot_price:.2f}</span>
""", unsafe_allow_html=True)

contract_df = fetch_option_contracts()
spot_price = fetch_spot_price(f"{EXCHANGE}|{SYMBOL}")
expiry_list = sorted(contract_df["expiry"].unique())
expiry = st.selectbox("Option Expiry", expiry_list, index=expiry_list.index(get_nearest_expiry(contract_df)))

st.sidebar.info("Strikes are fixed at 09:20 each day.")

if "strike_df" not in st.session_state or st.session_state.get("strikes_for_day") != (str(today), expiry):
    if now < strike_lock_time:
        st.info("Waiting for strike selection at 09:20 IST…")
        st.stop()
    else:
        sel_df = select_option_strikes(contract_df, spot_price, expiry, n=STRIKES_TO_PICK)
        filtered_df = sel_df[sel_df["strike_price"] % 100 == 0].sort_values(by=["instrument_type", "strike_price"])
        st.session_state["strike_df"] = filtered_df.copy()
        st.session_state["strikes_for_day"] = (str(today), expiry)

display_df = st.session_state["strike_df"]
keys_monitored = list(display_df["instrument_key"])

refresh_count = st_autorefresh(interval=5000, limit=1000, key="greeks_refresh")

# Initialize or incrementally append new poll values
if start_poll <= now <= end_poll:
    greek_data = poll_greeks_ltp(keys_monitored)
    timestamp = datetime.now(IST)

    if "greek_ts" not in st.session_state:
        st.session_state["greek_ts"] = []

    # Build new row
    row = {"timestamp": timestamp.isoformat()}
    for _, contract in display_df.iterrows():
        ikey = contract["instrument_key"]
        gd = greek_data.get(ikey, {})
        ltp = gd.get("ltp", float("nan"))
        row[f"{contract['instrument_type']}_{int(contract['strike_price'])}_ltp"] = ltp

        delta_val = gd.get("delta", None)
        if delta_val in [None, ""]:
            delta_val = fallback_compute(contract, spot_price, ltp).get("delta", float("nan"))
        if contract["instrument_type"] == "PE" and isinstance(delta_val, (int, float)) and not pd.isna(delta_val):
            delta_val = abs(delta_val)
        row[f"{contract['instrument_type']}_{int(contract['strike_price'])}_delta"] = delta_val

        for k in ["gamma", "theta"]:
            val = gd.get(k, None)
            if val in [None, ""]:
                val = fallback_compute(contract, spot_price, ltp).get(k, float("nan"))
            row[f"{contract['instrument_type']}_{int(contract['strike_price'])}_{k}"] = val

    st.session_state["greek_ts"].append(row)

else:
    st.info("Live polling active only between 09:20 and 15:20 IST.")

# Convert data to DataFrame for sending incremental updates
df = pd.DataFrame(st.session_state.get("greek_ts", []))
if df.empty:
    st.write("Waiting for data to accumulate…")
    st.stop()

# The metric list and their display names
greek_metrics = ["ltp", "delta", "gamma", "theta"]
names_for_caption = {
    "ltp": "Last Traded Price",
    "delta": "Delta",
    "gamma": "Gamma",
    "theta": "Theta"
}

# Maintain last sent timestamp index in session state
last_sent_idx = st.session_state.get("last_sent_idx", 0)

# Prepare new data points to send as JSON
new_data = df.iloc[last_sent_idx:]
st.session_state["last_sent_idx"] = len(df)

# Organize data per Greek and per Option Type for JS
def prepare_series_data(dataframe, metric):
    # Returns a list of dicts [{"name": col_name, "xs": [...], "ys": [...]}]
    series = []
    timestamp_list = list(dataframe["timestamp"])
    for col in dataframe.columns:
        if col.endswith(f"_{metric}"):
            ys = dataframe[col].tolist()
            # Deliver only x,y for new_data portion
            series.append({"name": col, "xs": timestamp_list, "ys": ys})
    return series

# Prepare all series JSON object for new points for all metrics
all_series = {}
for metric in greek_metrics:
    all_series[metric] = prepare_series_data(new_data, metric)

# Build JSON string for embedding
json_data = json.dumps(all_series)

# Create custom HTML + JS embedding plots with smooth incremental updates
html_string = f"""
<html>
<head>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  body {{ margin:0; }}
  .plot-container {{ width: 100%; height: 600px; }}
</style>
</head>
<body>
  <h3>Upstox Live Options Greeks - Incremental Plot</h3>
  <div style="display: flex; flex-wrap: wrap;">
    <div id="CE_ltp" class="plot-container"></div>
    <div id="PE_ltp" class="plot-container"></div>
    <div id="CE_delta" class="plot-container"></div>
    <div id="PE_delta" class="plot-container"></div>
    <div id="CE_gamma" class="plot-container"></div>
    <div id="PE_gamma" class="plot-container"></div>
    <div id="CE_theta" class="plot-container"></div>
    <div id="PE_theta" class="plot-container"></div>
  </div>
<script>
  let newPointData = {json_data};

  // For each metric, maintain full trace data
  const fullData = {{}};

  const plotDivs = {{
    "ltp": ["CE_ltp", "PE_ltp"],
    "delta": ["CE_delta", "PE_delta"],
    "gamma": ["CE_gamma", "PE_gamma"],
    "theta": ["CE_theta", "PE_theta"]
  }};

  // Initialize plots with empty traces
  for (const metric in plotDivs) {{
    plotDivs[metric].forEach(divId => {{
      const container = document.getElementById(divId);
      Plotly.newPlot(container, [], {{
        margin: {{ t: 30 }},
        xaxis: {{ title: "Time" }},
        yaxis: {{ title: metric.charAt(0).toUpperCase() + metric.slice(1) }},
        showlegend: true
      }});
      fullData[divId] = {{}};
    }});
  }}

  // Helper: split col name to extract CE or PE
  function getOptionType(col) {{
    return col.startsWith("CE_") ? "CE" : "PE";
  }}

  for (const metric in newPointData) {{
    const series = newPointData[metric];

    series.forEach(serie => {{
      const optionType = getOptionType(serie.name);
      const divId = optionType + "_" + metric;
      const container = document.getElementById(divId);

      if (!fullData[divId][serie.name]) {{
        // Init trace
        fullData[divId][serie.name] = {{
          x: [],
          y: [],
          mode: "lines",
          name: serie.name
        }};
        Plotly.addTraces(container, fullData[divId][serie.name]);
      }}

      // Append new points to trace
      const trace = fullData[divId][serie.name];
      // Compute new points from last length to current
      const lastLen = trace.x.length;
      const new_x = serie.xs.slice(lastLen);
      const new_y = serie.ys.slice(lastLen);

      trace.x = trace.x.concat(new_x);
      trace.y = trace.y.concat(new_y);

      Plotly.extendTraces(container, {{
        x: [new_x],
        y: [new_y]
      }}, [Object.keys(fullData[divId]).indexOf(serie.name)]);
    }});
  }}
</script>
</body>
</html>
"""

st.components.v1.html(html_string, height=850, scrolling=True)

# Provide CSV download button
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
