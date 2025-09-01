# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, time as dtime, timezone
import pytz
import time
import math
from math import log, sqrt, exp
from scipy.stats import norm
from scipy.optimize import brentq

# NOTE: This app uses the "upstox-api" Python SDK. Install via requirements.txt (below).
# The SDK's exact method names may vary; the implementation below targets the common Upstox SDK patterns.
# Put your API_KEY and API_SECRET into Streamlit Secrets as shown in instructions.

IST = pytz.timezone("Asia/Kolkata")
MARKET_REFRESH_TIME = dtime(9, 16)   # strike selection time
MARKET_START = dtime(9, 20)          # start streaming
MARKET_STOP  = dtime(15, 20)         # stop streaming

# -----------------------
# Black-Scholes functions
# -----------------------
def bs_price(option_type, S, K, T, r, sigma):
    """Black-Scholes price (European) for a call or put."""
    if T <= 0 or sigma <= 0:
        # At or past expiry: payoff
        return max(0.0, (S - K)) if option_type == "CE" else max(0.0, (K - S))
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "CE":
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def bs_greeks(option_type, S, K, T, r, sigma):
    """Return delta, gamma, theta (per day), vega (per 1 vol point)"""
    if T <= 0 or sigma <= 0:
        return 0.0, 0.0, 0.0, 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    if option_type == "CE":
        delta = norm.cdf(d1)
        theta = ( - (S * pdf_d1 * sigma) / (2 * math.sqrt(T))
                  - r * K * math.exp(-r * T) * norm.cdf(d2) )
    else:
        delta = -norm.cdf(-d1)
        theta = ( - (S * pdf_d1 * sigma) / (2 * math.sqrt(T))
                  + r * K * math.exp(-r * T) * norm.cdf(-d2) )
    gamma = pdf_d1 / (S * sigma * math.sqrt(T))
    vega = S * pdf_d1 * math.sqrt(T)
    # Convert theta to per-day (since above is per-year)
    theta_per_day = theta / 365.0
    return float(delta), float(gamma), float(theta_per_day), float(vega)

def implied_vol_from_price(option_type, market_price, S, K, T, r, bracket=(1e-4, 5.0)):
    """Estimate implied vol by solving bs_price(sigma) = market_price using brentq root finder."""
    def objective(sigma):
        return bs_price(option_type, S, K, T, r, sigma) - market_price
    # basic checks
    try:
        lower, upper = bracket
        # Ensure the signs differ
        f_low = objective(lower)
        f_high = objective(upper)
        if f_low * f_high > 0:
            # expand bracket heuristically
            lower = 1e-6
            upper = 10.0
        implied = brentq(objective, lower, upper, maxiter=200, xtol=1e-6)
        return float(implied)
    except Exception:
        return None

# -----------------------
# Helpers for timing
# -----------------------
def now_ist():
    return datetime.now(pytz.timezone("Asia/Kolkata"))

def time_in_window(start_time: dtime, stop_time: dtime):
    t = now_ist().time()
    return start_time <= t <= stop_time

# -----------------------
# Upstox client wrapper
# -----------------------
@st.cache_resource
def connect_upstox(access_token: str):
    """
    Connect to Upstox. This code expects the 'upstox_api' package (Upstox class).
    Put API_KEY and API_SECRET in Streamlit secrets as:
    [secrets]
    API_KEY = "your_key"
    API_SECRET = "your_secret"
    """
    try:
        from upstox_api.api import Upstox
    except Exception as e:
        st.error("Missing upstox_api package. Make sure requirements.txt contains 'upstox-api' and it's installed.")
        raise e

    api_key = st.secrets["API_KEY"]
    api_secret = st.secrets["API_SECRET"]
    u = Upstox(api_key, api_secret)
    # set access token that the user pastes
    u.set_access_token(access_token)
    return u

# -----------------------
# Fetch expiries & contracts
# -----------------------
def fetch_expiry_list(u, symbol_root="NIFTY"):
    """
    Returns sorted list of next expiries (as date objects). Uses master contracts from Upstox.
    """
    try:
        master = u.get_master_contract('NSE_FO')
    except Exception as e:
        st.error("Failed to get master contract list from Upstox: " + str(e))
        return []
    # master may be a dict: token -> object with .symbol and .expiry attributes
    expiries = set()
    for _token, inst in master.items():
        # best-effort matching: symbol contains NIFTY (varies by SDK)
        sym = getattr(inst, "symbol", "") or getattr(inst, "instrument", "") or ""
        if symbol_root in sym.upper():
            exp = getattr(inst, "expiry", None)
            if exp:
                # expiry might be string 'YYYY-MM-DD' or date
                if isinstance(exp, str):
                    try:
                        exp_dt = datetime.fromisoformat(exp).date()
                    except Exception:
                        continue
                else:
                    exp_dt = exp
                expiries.add(exp_dt)
    ex_sorted = sorted(list(expiries))
    return ex_sorted[:6]  # return more so UI shows choices; we'll show next 4

def pick_contracts_for_day(u, expiry_date, strikes_to_pick=5, rounding=50, symbol_root="NIFTY"):
    """
    At strike-selection time: fetch spot, round to nearest Strike (50), pick strikes
    Returns a list of contract dicts: {strike, option_type('CE'/'PE'), token, symbol}
    """
    # fetch spot (NIFTY index LTP)
    try:
        # some SDKs accept 'NSE_INDEX|Nifty 50' key; we try common keys
        for key in ["NSE_INDEX|Nifty 50", "NSE_INDEX|NIFTY 50", "NSE:NIFTY 50"]:
            try:
                l = u.get_live_feed(key, "LTP")
                spot = float(l.get("ltp"))
                break
            except Exception:
                spot = None
        if spot is None:
            # fallback: find an index instrument in master and call get_live_feed on its token
            master = u.get_master_contract('NSE_INDEX')
            for token, inst in master.items():
                sym = getattr(inst, "symbol", "") or ""
                if "NIFTY" in sym.upper():
                    try:
                        l = u.get_live_feed(token, "LTP")
                        spot = float(l.get("ltp"))
                        break
                    except Exception:
                        continue
        if spot is None:
            raise RuntimeError("Could not fetch NIFTY spot LTP from Upstox.")
    except Exception as e:
        raise RuntimeError("Error fetching spot: " + str(e))

    # round to nearest strike (50)
    atm_strike = int(round(spot / rounding) * rounding)

    # collect option contracts for that expiry from master
    master = u.get_master_contract('NSE_FO')
    # build a mapping strike -> tokens
    strikes_map = {}
    for token, inst in master.items():
        sym = getattr(inst, "symbol", "") or ""
        exp = getattr(inst, "expiry", None)
        strike = getattr(inst, "strike_price", None)
        itype = getattr(inst, "instrument_type", None) or getattr(inst, "option_type", None)
        if not (strike and exp):
            continue
        # normalize expiry
        if isinstance(exp, str):
            try:
                exp_dt = datetime.fromisoformat(exp).date()
            except Exception:
                continue
        else:
            exp_dt = exp
        if exp_dt != expiry_date:
            continue
        # ensure symbol_root present
        if symbol_root not in sym.upper():
            continue
        strikes_map.setdefault(strike, {})[itype] = {"token": token, "symbol": sym}

    # sorted strikes list
    strikes_sorted = sorted(strikes_map.keys())
    if atm_strike not in strikes_sorted:
        # pick closest available
        atm_strike = min(strikes_sorted, key=lambda x: abs(x - atm_strike))

    idx = strikes_sorted.index(atm_strike)
    low_idx = max(0, idx - strikes_to_pick)
    high_idx = min(len(strikes_sorted) - 1, idx + strikes_to_pick)
    selected = []
    # pick 5 below (ITM for Calls) and 5 above (OTM for Calls), but return all option contracts across CE/PE
    chosen_strikes = strikes_sorted[low_idx: high_idx + 1]  # includes ATM
    # We'll use exactly 5 ITM + 5 OTM where possible: pick 5 below and 5 above excluding ATM if needed
    # Determine 5 below and 5 above relative to ATM index
    below = strikes_sorted[max(0, idx - strikes_to_pick): idx]
    above = strikes_sorted[idx+1: idx+1+strikes_to_pick]
    # For robustness, combine
    final_strikes = list(below[-strikes_to_pick:]) + [strikes_sorted[idx]] + list(above[:strikes_to_pick])
    # Now create contract entries (CE and PE) for each selected strike, but we will track only 5 ITM & 5 OTM logic in naming
    for s in final_strikes:
        entry = strikes_map.get(s, {})
        ce = entry.get("CE") or entry.get("CALL") or entry.get("C")
        pe = entry.get("PE") or entry.get("PUT") or entry.get("P")
        if ce:
            selected.append({"strike": s, "type": "CE", "token": ce["token"], "symbol": ce["symbol"]})
        if pe:
            selected.append({"strike": s, "type": "PE", "token": pe["token"], "symbol": pe["symbol"]})
    # Return spot and selected contracts and the atm strike for record
    return spot, int(atm_strike), selected

# -----------------------
# UI
# -----------------------
st.set_page_config(layout="wide", page_title="NIFTY Greeks - Streamlit")
st.title("📈 NIFTY Option Greeks — Live (Streamlit)")

st.markdown("""
**How to use**
1. Add `API_KEY` and `API_SECRET` to Streamlit Secrets.  
2. Paste your **daily Upstox Access Token** below and press Enter.  
3. App will automatically fetch expiries; at **09:16 IST** it will pick today's strikes and lock them.  
4. Live streaming begins at **09:20 IST** and ends at **15:20 IST**.
""")

# Access token input (user pastes daily)
access_token = st.text_input("🔑 Paste Upstox ACCESS_TOKEN (daily)", type="password")

if not access_token:
    st.warning("Enter your Upstox access token to proceed.")
    st.stop()

# Connect
try:
    upstox_client = connect_upstox(access_token)
except Exception as e:
    st.error("Could not connect to Upstox: " + str(e))
    st.stop()

# Fetch expiries
expiries = fetch_expiry_list(upstox_client, symbol_root="NIFTY")
if not expiries:
    st.error("Could not fetch expiries. Check your API access and master contracts.")
    st.stop()

# Show expiry choices (next 4)
expiry_options = expiries[:4] if len(expiries) >= 1 else expiries
expiry_choice = st.selectbox("Choose expiry (default = nearest)", expiry_options, index=0)

# Session state to hold daily selected contracts
if "contracts" not in st.session_state:
    st.session_state.contracts = None
    st.session_state.refresh_date = None
    st.session_state.atm = None
    st.session_state.spot = None

# If it's past 09:16 IST and we haven't refreshed today -> refresh now
current_ist = now_ist()
if current_ist.time() >= MARKET_REFRESH_TIME and st.session_state.refresh_date != current_ist.date():
    try:
        spot, atm_strike, sel_contracts = pick_contracts_for_day(upstox_client, expiry_choice, strikes_to_pick=5)
        st.session_state.contracts = sel_contracts
        st.session_state.refresh_date = current_ist.date()
        st.session_state.atm = atm_strike
        st.session_state.spot = spot
        st.success(f"Selected {len(sel_contracts)} option contracts for the day (ATM ≈ {atm_strike}, spot {spot:.2f}).")
    except Exception as e:
        st.error("Failed to pick contracts at 09:16: " + str(e))
        st.stop()
else:
    # show what was selected previously (if any)
    if st.session_state.contracts:
        st.info(f"Contracts locked for {st.session_state.refresh_date} | ATM: {st.session_state.atm} | Spot (when locked): {st.session_state.spot:.2f}")
    else:
        st.info("Contracts will be auto-selected at 09:16 IST. You can keep this page open.")

# Stop if contracts aren't ready yet
if not st.session_state.contracts:
    st.stop()

# Prepare display placeholders
placeholder_top = st.empty()
placeholder_charts = st.empty()
placeholder_table = st.empty()

# Prepare data storage for plotting time series
if "history" not in st.session_state:
    st.session_state.history = {}  # key = token, value = list of {ts, ltp, delta, gamma, theta, vega}
for c in st.session_state.contracts:
    if c["token"] not in st.session_state.history:
        st.session_state.history[c["token"]] = []

# Market loop: runs while in streaming window
st.info(f"Live updates will run between {MARKET_START.strftime('%H:%M')} and {MARKET_STOP.strftime('%H:%M')} IST.")

# Build labels for display order
labels = [f"{c['strike']}{c['type']}" for c in st.session_state.contracts]

# Live update loop with Streamlit-friendly pattern
# We'll run updates while within market hours; Streamlit will keep the session alive when the page is open.
while True:
    if not time_in_window(MARKET_START, MARKET_STOP):
        placeholder_top.info("Outside streaming window. Waiting until market opens (09:20 IST) or until next day.")
        time.sleep(5)
        # If past stop time, break loop so app doesn't keep hammering API unnecessarily.
        if now_ist().time() > MARKET_STOP:
            placeholder_top.warning("Market window over for today. Streaming stopped.")
            break
        continue

    try:
        # For each contract, fetch LTP and implied vol (if available)
        df_rows = []
        for c in st.session_state.contracts:
            token = c["token"]
            strike = c["strike"]
            otype = c["type"]  # 'CE' or 'PE'
            symbol = c.get("symbol", f"{strike}{otype}")

            ltp = None
            iv  = None
            market_price = None
            # Try to fetch live feed; different SDKs return different shapes. Wrap in try/except
            try:
                feed = upstox_client.get_live_feed(token, "LTP")  # returns dict-like
                # common keys: 'ltp', 'implied_volatility', 'last_price'
                ltp = float(feed.get("ltp") or feed.get("last_price") or 0.0)
                iv  = feed.get("implied_volatility") or feed.get("iv") or None
                market_price = ltp
            except Exception:
                # fallback to trying with token as int/string
                try:
                    feed = upstox_client.get_live_feed(str(token), "LTP")
                    ltp = float(feed.get("ltp") or feed.get("last_price") or 0.0)
                    iv  = feed.get("implied_volatility") or feed.get("iv") or None
                    market_price = ltp
                except Exception:
                    ltp = None

            # If no LTP, set NaN
            if ltp is None:
                ltp = float("nan")

            # Compute time to expiry in years (precise seconds)
            if isinstance(expiry_choice, datetime):
                expiry_date_dt = expiry_choice
            else:
                # expiry_choice is date object
                expiry_date_dt = expiry_choice
            # expiry at market close time (assume 15:30 IST on expiry date)
            expiry_dt = datetime.combine(expiry_date_dt, dtime(15, 30)).replace(tzinfo=IST)
            nowdt = now_ist()
            seconds = max((expiry_dt - nowdt).total_seconds(), 0.0)
            T = seconds / (365.0 * 24 * 3600)  # fraction of year

            # If IV present from feed, use it; else try to estimate from market price (LTP)
            iv_used = None
            if iv not in (None, 0, "", []):
                try:
                    iv_used = float(iv)
                    # if IV given in percent (e.g. 12.5), convert to decimal
                    if iv_used > 5:  # heuristic: >5 likely in percent
                        iv_used = iv_used / 100.0
                except Exception:
                    iv_used = None

            if iv_used is None and (market_price is not None and not math.isnan(market_price)):
                # estimate iv numerically from market_price
                try:
                    est_iv = implied_vol_from_price(otype, market_price, st.session_state.spot, strike, T, 0.05)
                    iv_used = est_iv if est_iv is not None else 0.20
                except Exception:
                    iv_used = 0.20  # fallback default
            if iv_used is None:
                iv_used = 0.20

            delta, gamma, theta, vega = bs_greeks(otype, st.session_state.spot, strike, T, r=0.05, sigma=iv_used)

            # store
            row = {
                "token": token,
                "label": f"{strike}{otype}",
                "strike": strike,
                "type": otype,
                "ltp": ltp,
                "iv": iv_used,
                "delta": delta,
                "gamma": gamma,
                "theta": theta,
                "vega": vega,
                "ts": now_ist()
            }
            df_rows.append(row)
            # append to history
            st.session_state.history[token].append(row)

        # Build dataframe for current snapshot
        snapshot_df = pd.DataFrame(df_rows).sort_values(by=["strike", "type"])

        # Plotly time-series: for each greek, one subplot; each contract is its own line
        figs = []
        greeks = ["delta", "gamma", "theta", "vega"]
        fig = make_subplots = None
        # We'll build a single figure with 4 rows
        fig = make_plot_for_greeks = go.Figure()
        # For clarity, create one figure with 4 subplots using row-wise annotation via domains
        # Simpler approach: create 4 stacked figures vertically
        rows = []
        for g in greeks:
            fig_g = go.Figure()
            for c in st.session_state.contracts:
                token = c["token"]
                hist = st.session_state.history.get(token, [])
                if not hist:
                    continue
                xs = [h["ts"] for h in hist]
                ys = [h[g] for h in hist]
                label = f"{c['strike']}{c['type']}"
                fig_g.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=label, hovertemplate='%{y:.4f}<br>%{x}'))
            fig_g.update_layout(title=g.capitalize(), showlegend=True, xaxis_title="Time (IST)")
            rows.append(fig_g)

        # Render top info, table, and charts
        with placeholder_top.container():
            st.metric("ATM (locked)", st.session_state.atm)
            st.metric("Spot (when locked)", f"{st.session_state.spot:.2f}")
            st.write(f"Expiry chosen: {expiry_choice}")

        with placeholder_table.container():
            st.subheader("Current snapshot")
            st.dataframe(snapshot_df[["label", "ltp", "iv", "delta", "gamma", "theta", "vega"]].set_index("label"))

        with placeholder_charts.container():
            st.subheader("Live Greeks Time Series")
            for fig_g in rows:
                st.plotly_chart(fig_g, use_container_width=True)

    except Exception as e:
        st.error("Error during live update: " + str(e))

    # Pause approx 1 second between updates
    time.sleep(1)

# End of streaming loop
st.info("Streaming finished for today. Refresh tomorrow for new strikes.")
