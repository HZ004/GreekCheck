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

# --- Your other helper functions remain unchanged ---

st.set_page_config(layout="wide")
st.title("Upstox Live Options Greeks Dashboard")

# --- User Input ---
upstox_token = st.text_input(
    "Paste Upstox Access Token", type="password",
    help="Token expires daily, paste every morning."
)

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

# --- Fetch contracts & spot price ---
contract_df = fetch_option_contracts()
spot_price = fetch_spot_price(f"{EXCHANGE}|{SYMBOL}")

# --- Expiry selection ---
expiry_list = sorted(contract_df["expiry"].unique())
nearest_expiry = get_nearest_expiry(contract_df)
expiry = st.selectbox("Option Expiry", expiry_list,
                      index=expiry_list.index(nearest_expiry))

# --- Select strikes if not yet selected or day changed ---
IST = ZoneInfo("Asia/Kolkata")
now = datetime.now(IST)
today = now.date()
strike_lock_time = datetime.combine(today, time(9, 16), IST)

if ("strike_df" not in st.session_state
        or st.session_state.get("strikes_for_day") != (str(today), expiry)):
    if now < strike_lock_time:
        st.info("Waiting for strike selection at 09:16 IST…")
        st.stop()
    else:
        sel_df = select_option_strikes(contract_df, spot_price, expiry, n=STRIKES_TO_PICK)

        # Filter strikes divisible by 100 plus ATM forcibly included
        filtered_df = sel_df[sel_df["strike_price"] % 100 == 0].sort_values(
            by=["instrument_type", "strike_price"]
        )
        st.session_state["strike_df"] = filtered_df.copy()
        st.session_state["strikes_for_day"] = (str(today), expiry)

display_df = st.session_state["strike_df"]

# Define keys_monitored here!
keys_monitored = list(display_df["instrument_key"])

# --- Live polling setup ---
start_poll = datetime.combine(today, time(9, 20), IST)
end_poll = datetime.combine(today, time(15, 20), IST)

refresh_count = st_autorefresh(interval=5000, limit=1000, key="greeks_refresh")
table_placeholder = st.empty()

if start_poll <= now <= end_poll:
    datalist = st.session_state.get("greek_ts", [])

    # Poll Greeks once on each refresh
    greek_data = poll_greeks_ltp(keys_monitored)
    timestamp = datetime.now(IST)

    row = {"timestamp": timestamp}
    for _, contract in display_df.iterrows():
        ikey = contract["instrument_key"]
        gd = greek_data.get(ikey, {})
        ltp = gd.get("ltp", float("nan"))
        row.update(
            {
                f"{contract['instrument_type']}_{int(contract['strike_price'])}_{k}": (
                    gd.get(k)
                    if gd.get(k) not in [None, ""]
                    else fallback_compute(contract, spot_price, ltp).get(k, float("nan"))
                )
                for k in ["delta", "gamma", "vega", "theta", "rho"]
            }
        )

    datalist.append(row)
    st.session_state["greek_ts"] = datalist

    df = pd.DataFrame(datalist)

    # Display styled strikes table
    styled_df = display_df[["instrument_type", "strike_price", "expiry"]].copy()
    styled_df["strike_price"] = styled_df["strike_price"].astype(int)
    styled_df["expiry"] = pd.to_datetime(styled_df["expiry"]).dt.strftime("%Y-%m-%d")
    styled_df = styled_df.rename(
        columns={
            "instrument_type": "Option Type",
            "strike_price": "Strike Price",
            "expiry": "Expiry Date",
        }
    )

    def highlight_option_type(row):
        if row["Option Type"] == "CE":
            return ["background-color: #d0e7ff"] * len(row)
        elif row["Option Type"] == "PE":
            return ["background-color: #ffd6d6"] * len(row)
        else:
            return [""] * len(row)

    table_placeholder.dataframe(
        styled_df.style.apply(highlight_option_type, axis=1), height=400, use_container_width=True
    )

    # Plot Greeks 2x2
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

    # Download CSV Button
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
