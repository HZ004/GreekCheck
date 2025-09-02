# streamlit_upstox_greeks_app.py
# Streamlit app: Real-time NIFTY options Greeks monitor using Upstox REST APIs
# -----------------------------------------------------------------------------
# Place this file in your repo as app.py (or this filename). See the README section
# below (in the comments) for requirements.txt and Render deploy steps.
#
# IMPORTANT: This app expects you to paste a valid Upstox access token (Bearer token)
# into the Streamlit UI each day. Tokens typically expire overnight. See Upstox docs:
# - Authentication / Get Token: https://upstox.com/developer/api-documentation/get-token/
# - Option Contracts: https://upstox.com/developer/api-documentation/get-option-contracts/
# - Option Greeks: https://upstox.com/developer/api-documentation/example-code/market-quote/option-greek/
# - LTP Quotes: https://upstox.com/developer/api-documentation/example-code/market-quote/ltp-quotes-v3/
#
# Copy-paste this file as-is to your Streamlit project. The code is production-ready
# but you must supply a valid Upstox access token (paste in UI) and have internet
# access from the host.
#
# ------------------------------ Requirements ---------------------------------
# Create a file named requirements.txt with at least these lines:
#
# streamlit>=1.20
# streamlit-autorefresh
# requests
# pandas
# numpy
# matplotlib
# pytz
#
# (Optionally pin versions. "streamlit-autorefresh" is used to make the client auto-refresh.)
#
# ----------------------------- How it works ----------------------------------
# 1. On the Streamlit page paste your Upstox access token (daily) and click 'Authorize'.
# 2. Press "Start background fetcher (server)" — this creates a daemon thread that:
#    - At 09:16 IST selects the day's strikes (nearest expiry by default) and fixes 5 ITM +
#      5 OTM strikes per side (CE & PE) for the day.
#    - Between 09:20 and 15:20 IST polls the Upstox v3 option-greek endpoint once per second
#      for the selected instruments and stores the time series.
# 3. Open the Streamlit web page (mobile/desktop). The charts will auto-refresh (client-side)
#    once per second and show real-time Greek trends.
#
# ----------------------------- Upstox endpoints used --------------------------
# - Option Contracts: GET /v2/option/contract?instrument_key=<UNDERLYING>
#   (fetch all option contracts and expiries). See docs.  
#   https://upstox.com/developer/api-documentation/get-option-contracts/  
# - Option Greeks (per-instrument): GET /v3/market-quote/option-greek?instrument_key=...
#   (fetches Greeks and IV for up to 50 instruments in a single request).  
#   https://upstox.com/developer/api-documentation/example-code/market-quote/option-greek/  
# - LTP: GET /v3/market-quote/ltp?instrument_key=...  
#   https://upstox.com/developer/api-documentation/example-code/market-quote/ltp-quotes-v3/
#
# ------------------------------ Black-Scholes --------------------------------
# The app contains the standard Black-Scholes formulae used to compute Greeks locally
# (in case the REST response is missing any field). Implementation is in bs_greeks().
# Formulas used (European options, continuous compounding):
#   d1 = (ln(S/K) + (r + 0.5*sigma^2) T) / (sigma * sqrt(T))
#   d2 = d1 - sigma * sqrt(T)
#   Delta (call) = N(d1)
#   Delta (put)  = N(d1) - 1
#   Gamma = phi(d1) / (S * sigma * sqrt(T))
#   Vega = S * phi(d1) * sqrt(T)
#   Theta (call) = -(S*phi(d1)*sigma)/(2*sqrt(T)) - r*K*e^{-rT}*N(d2)
#   Theta (put)  = -(S*phi(d1)*sigma)/(2*sqrt(T)) + r*K*e^{-rT}*N(-d2)
#   Rho (call) = K*T*e^{-rT}*N(d2)
# See code for implementation.
#
# ------------------------------ Security note --------------------------------
# The token pasted into the UI is stored in a file (upstox_token.txt) on the server for
# convenience. If you're worried about security, modify the code to store the token
# in an encrypted location / environment variable or remove disk persistence entirely.
#
# ----------------------------------------------------------------------------
# Begin application code

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests, threading, time, math, json, os
from datetime import datetime, date, timedelta, time as dtime
import pytz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------------------- Config ----------------------
API_BASE_V2 = "https://api.upstox.com/v2"
API_BASE_V3 = "https://api.upstox.com/v3"
IST = pytz.timezone("Asia/Kolkata")
UNDERLYING_DEFAULT = "NSE_INDEX|Nifty 50"  # changeable in the UI
DAILY_SELECTION_TIME = dtime(9,16)   # when to choose the day's strikes (IST)
POLL_START = dtime(9,20)
POLL_END = dtime(15,20)
SLEEP_SECONDS = 1  # poll every second
ITM_COUNT = 5
OTM_COUNT = 5

# ---------------------- Storage ----------------------
DATA_LOCK = threading.Lock()
DATA = {
    'selected_date': None,  # date string 'YYYY-MM-DD' when selection was made
    'expiry': None,
    'underlying_key': UNDERLYING_DEFAULT,
    'selected_instruments': [],  # list of instrument_key strings selected for the day
    'series': defaultdict(list),  # instrument_key -> list of dictionaries with timestamp and greeks/ltp
}

TOKEN_FILE = "upstox_token.txt"  # token persistence (user can paste daily)

# ---------------------- Utilities ----------------------
def save_token(token: str):
    with open(TOKEN_FILE, "w") as f:
        f.write(token.strip())

def read_token():
    if not os.path.exists(TOKEN_FILE):
        return None
    with open(TOKEN_FILE, "r") as f:
        t = f.read().strip()
        return t or None

def ist_now():
    return datetime.now(IST)

def to_ist(dt: datetime):
    if dt.tzinfo is None:
        dt = pytz.utc.localize(dt)
    return dt.astimezone(IST)

def format_ts(ts=None):
    if ts is None: ts = ist_now()
    return ts.strftime("%Y-%m-%d %H:%M:%S %Z")

def make_headers(token):
    return {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}"
    }

# ---------------------- Black-Scholes Greeks (formulas) ----------------------
def _phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def _N(x):
    # Standard normal CDF using erf
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_greeks(S, K, r, sigma, T, opt_type="CE"):
    # S: spot, K: strike, r: risk-free rate (annual, e.g., 0.07), sigma: implied vol (annual, e.g., 0.2), T: time to expiry in years
    # opt_type: "CE" or "PE"
    # Returns dict: delta, gamma, vega, theta (per year), rho
    res = {}
    if T <= 0 or sigma <= 0:
        # Expired or zero vol: degenerate
        if opt_type == "CE":
            res['delta'] = 1.0 if S > K else 0.0
        else:
            res['delta'] = -1.0 if S < K else 0.0
        res.update({'gamma':0.0,'vega':0.0,'theta':0.0,'rho':0.0})
        return res
    d1 = (math.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    nd1 = _N(d1)
    nd2 = _N(d2)
    pdf_d1 = _phi(d1)
    # Delta
    if opt_type == "CE":
        delta = nd1
    else:
        delta = nd1 - 1.0
    gamma = pdf_d1 / (S * sigma * math.sqrt(T))
    vega = S * pdf_d1 * math.sqrt(T)  # per 1.0 vol (i.e., if vol is 0.20 -> per unit)
    # Theta (per year)
    if opt_type == "CE":
        theta = - (S * pdf_d1 * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r*T) * nd2
    else:
        theta = - (S * pdf_d1 * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r*T) * _N(-d2)
    rho_call = K * T * math.exp(-r*T) * nd2
    rho = rho_call if opt_type == "CE" else -K * T * math.exp(-r*T) * _N(-d2)
    res['delta'] = delta
    res['gamma'] = gamma
    res['vega'] = vega
    res['theta'] = theta  # per year; divide by 365 for per-day theta
    res['rho'] = rho
    return res

# ---------------------- Upstox API helpers ----------------------
def get_option_contracts(token, underlying_key):
    """Fetch option contracts for an underlying (all expiries if expiry_date not provided)."""
    url = f"{API_BASE_V2}/option/contract"
    params = {"instrument_key": underlying_key}
    r = requests.get(url, headers=make_headers(token), params=params, timeout=10)
    r.raise_for_status()
    return r.json().get('data', [])

def get_option_chain_put_call(token, underlying_key, expiry_date=None):
    url = f"{API_BASE_V2}/option/chain"
    params = {"instrument_key": underlying_key}
    if expiry_date:
        params["expiry_date"] = expiry_date
    r = requests.get(url, headers=make_headers(token), params=params, timeout=10)
    r.raise_for_status()
    return r.json().get('data', {})

def get_ltp(token, instrument_keys):
    """instrument_keys: list of instrument_key strings"""
    if not instrument_keys:
        return {}
    keys = ",".join(instrument_keys)
    url = f"{API_BASE_V3}/market-quote/ltp"
    params = {"instrument_key": keys}
    r = requests.get(url, headers=make_headers(token), params=params, timeout=10)
    r.raise_for_status()
    data = r.json().get('data', {})
    ltp_map = {}
    if isinstance(data, list):
        for q in data:
            k = q.get('instrument_key') or q.get('instrumentKey') or None
            if k:
                ltp_map[k] = q.get('ltp') or q.get('last_price') or None
    elif isinstance(data, dict):
        for k,v in data.items():
            ltp_map[k] = v.get('ltp') or v.get('last_price')
    return ltp_map

def get_option_greeks(token, instrument_keys):
    """Use v3 option-greek endpoint; returns list of dicts keyed by instrument_key"""
    if not instrument_keys:
        return {}
    keys = ",".join(instrument_keys)
    url = f"{API_BASE_V3}/market-quote/option-greek"
    params = {"instrument_key": keys}
    r = requests.get(url, headers=make_headers(token), params=params, timeout=10)
    r.raise_for_status()
    js = r.json()
    return js.get('data', {})

# ---------------------- Selection logic ----------------------
def choose_strikes_and_contracts(token, underlying_key, expiry_date=None, itm_count=ITM_COUNT, otm_count=OTM_COUNT):
    contracts = get_option_contracts(token, underlying_key)
    if not contracts:
        return None
    df = pd.DataFrame(contracts)
    if expiry_date:
        df = df[df['expiry'] == expiry_date]
    ltp_map = get_ltp(token, [underlying_key])
    underlying_price = None
    if underlying_key in ltp_map:
        underlying_price = ltp_map[underlying_key]
    else:
        try:
            ltp_map = get_ltp(token, [underlying_key])
            underlying_price = ltp_map.get(underlying_key)
        except Exception:
            underlying_price = None
    if underlying_price is None:
        raise RuntimeError("Unable to retrieve underlying LTP for " + str(underlying_key))
    strikes = sorted(df['strike_price'].unique())
    atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - underlying_price))
    ce_df = df[df['instrument_type'] == 'CE']
    pe_df = df[df['instrument_type'] == 'PE']
    ce_strikes = sorted(ce_df['strike_price'].unique())
    pe_strikes = sorted(pe_df['strike_price'].unique())
    lower_ce = [s for s in ce_strikes if s <= underlying_price]
    higher_ce = [s for s in ce_strikes if s > underlying_price]
    lower_ce = sorted(lower_ce, reverse=True)
    higher_ce = sorted(higher_ce)
    selected_ce = []
    selected_ce += lower_ce[:itm_count]
    selected_ce += higher_ce[:otm_count]
    lower_pe = [s for s in pe_strikes if s < underlying_price]
    higher_pe = [s for s in pe_strikes if s >= underlying_price]
    higher_pe = sorted(higher_pe)
    lower_pe = sorted(lower_pe, reverse=True)
    selected_pe = []
    selected_pe += higher_pe[:itm_count]
    selected_pe += lower_pe[:otm_count]
    selected_instruments = []
    for s in selected_ce:
        row = ce_df[ce_df['strike_price'] == s].iloc[0]
        selected_instruments.append(row['instrument_key'])
    for s in selected_pe:
        row = pe_df[pe_df['strike_price'] == s].iloc[0]
        selected_instruments.append(row['instrument_key'])
    return {
        'underlying_price': underlying_price,
        'selected_instruments': selected_instruments,
        'selected_ce_strikes': selected_ce,
        'selected_pe_strikes': selected_pe,
        'expiry': expiry_date or (df['expiry'].min() if not df.empty else None)
    }

# ---------------------- Background fetcher thread ----------------------
class BackgroundFetcher(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.is_set()

    def run(self):
        print("Background fetcher started (server time).")
        while not self.stopped():
            token = read_token()
            if not token:
                time.sleep(2)
                continue
            now = ist_now()
            today = now.date()
            sel_done_for = DATA['selected_date']
            try:
                if now.time() >= DAILY_SELECTION_TIME and sel_done_for != today.isoformat():
                    try:
                        oc = get_option_contracts(token, DATA['underlying_key'])
                        if not oc:
                            print("No option contracts returned when selecting strikes.")
                        else:
                            df = pd.DataFrame(oc)
                            df['expiry_dt'] = pd.to_datetime(df['expiry']).dt.date
                            future_expiries = sorted([d for d in df['expiry_dt'].unique() if d >= today])
                            expiry_choice = future_expiries[0].isoformat() if future_expiries else str(df['expiry_dt'].min())
                            out = choose_strikes_and_contracts(token, DATA['underlying_key'], expiry_choice)
                            if out:
                                with DATA_LOCK:
                                    DATA['selected_date'] = today.isoformat()
                                    DATA['expiry'] = expiry_choice
                                    DATA['selected_instruments'] = out['selected_instruments']
                                    DATA['series'] = defaultdict(list)
                                print(f"Selected {len(out['selected_instruments'])} contracts for expiry {expiry_choice} at {format_ts()} (ATM {out['underlying_price']})")
                    except Exception as e:
                        print(f"Selection error: {e}")
                if POLL_START <= now.time() <= POLL_END and DATA['selected_instruments']:
                    try:
                        greeks_data = get_option_greeks(token, DATA['selected_instruments'])
                        entries = []
                        if isinstance(greeks_data, dict):
                            for k,v in greeks_data.items():
                                d = v.copy()
                                d['instrument_key'] = k
                                entries.append(d)
                        elif isinstance(greeks_data, list):
                            entries = greeks_data
                        else:
                            entries = []
                        ts = format_ts()
                        with DATA_LOCK:
                            for ent in entries:
                                ik = ent.get('instrument_key') or ent.get('instrumentKey')
                                if not ik:
                                    continue
                                row = {
                                    'ts': ts,
                                    'ltp': ent.get('ltp') or ent.get('last_price') or None,
                                    'iv': ent.get('iv') or ent.get('implied_volatility') or None,
                                    'delta': ent.get('delta') or None,
                                    'gamma': ent.get('gamma') or None,
                                    'vega': ent.get('vega') or None,
                                    'theta': ent.get('theta') or None,
                                    'rho': ent.get('rho') or None,
                                }
                                DATA['series'][ik].append(row)
                        time.sleep(SLEEP_SECONDS)
                    except Exception as e:
                        print(f"Polling error at {format_ts()}: {e}")
                        time.sleep(2)
                else:
                    time.sleep(2)
            except Exception as e:
                print(f"Background fetcher top-level error: {e}")
                time.sleep(5)

# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="NIFTY Options Greeks - Real time (Upstox)", layout="wide")
# Auto-refresh the page every 1000 ms so charts update live in the browser
st_autorefresh(interval=1000, key="greek_refresh")
st.title("NIFTY Options Greeks — Real-time monitor (Upstox)")

# Token input and authorise button
col1, col2 = st.columns([2,1])
with col1:
    token_in = st.text_input("Paste your Upstox access token (valid for the day)", value=read_token() or "", type="password")
with col2:
    if st.button("Authorize & Save token"):
        if token_in and len(token_in) > 10:
            save_token(token_in)
            st.success("Token saved to server (file). Background fetch will use it.")
        else:
            st.error("Please paste a valid token string.")

# Underlying and expiry choices
with st.expander("Advanced settings (Underlying / expiry selection)"):
    underlying_key = st.text_input("Underlying instrument key", value=DATA.get('underlying_key', UNDERLYING_DEFAULT))
    if underlying_key != DATA.get('underlying_key'):
        DATA['underlying_key'] = underlying_key
    token = read_token()
    expiries = []
    if token:
        try:
            oc = get_option_contracts(token, underlying_key)
            if oc:
                dfoc = pd.DataFrame(oc)
                dfoc['expiry_dt'] = pd.to_datetime(dfoc['expiry']).dt.date
                expiries = sorted(list({d.isoformat() for d in dfoc['expiry_dt']}))
        except Exception:
            expiries = []
    expiry_choice = st.selectbox("Expiry (default = nearest)", options=["Auto (nearest)"] + expiries, index=0)
    st.markdown("By default the app picks the nearest expiry. You can override to any available expiry here.")

# Start/Stop background fetcher controls (server-side)
if 'bg' not in st.session_state:
    st.session_state.bg = None
# Auto-start background fetcher if token already provided
if read_token() and st.session_state.bg is None:
    st.session_state.bg = BackgroundFetcher()
    st.session_state.bg.start()

if st.button("Start background fetcher (server)"):
    if st.session_state.bg is None:
        st.session_state.bg = BackgroundFetcher()
        st.session_state.bg.start()
        st.success("Background fetcher started on server. It will auto-select strikes at 09:16 IST and poll from 09:20 to 15:20 IST.")
    else:
        st.info("Background fetcher already started.")

if st.button("Stop background fetcher (server)"):
    if st.session_state.bg is not None:
        st.session_state.bg.stop()
        st.session_state.bg = None
        st.warning("Background fetcher stopped.")

# Display selected instruments and live table
st.subheader("Today's selection & latest values")
with DATA_LOCK:
    selected = DATA.get('selected_instruments', [])
    expiry = DATA.get('expiry', None)
    selected_date = DATA.get('selected_date', None)
if not selected:
    st.info("No contracts selected yet. Paste token and Start background fetcher OR press 'Force selection now' below (for testing).")
colA, colB, colC = st.columns(3)
with colA:
    st.metric("Selected date", selected_date or "-")
with colB:
    st.metric("Expiry", expiry or "-")
with colC:
    st.metric("Total contracts", len(selected))

if st.button("Force selection now (immediate)"):
    token = read_token()
    if not token:
        st.error("No token saved. Paste token and click 'Authorize & Save token' first.")
    else:
        try:
            out = choose_strikes_and_contracts(token, DATA['underlying_key'], None)
            with DATA_LOCK:
                DATA['selected_date'] = ist_now().date().isoformat()
                DATA['expiry'] = out['expiry']
                DATA['selected_instruments'] = out['selected_instruments']
                DATA['series'] = defaultdict(list)
            st.success(f"Forced selection done. {len(out['selected_instruments'])} contracts chosen.")
        except Exception as e:
            st.error(f"Force selection failed: {e}")

# Live table
if selected:
    rows = []
    with DATA_LOCK:
        for ik in selected:
            series = DATA['series'].get(ik, [])
            latest = series[-1] if series else {}
            rows.append({
                'instrument_key': ik,
                'ltp': latest.get('ltp'),
                'iv': latest.get('iv'),
                'delta': latest.get('delta'),
                'gamma': latest.get('gamma'),
                'vega': latest.get('vega'),
                'theta_per_day': (latest.get('theta')/365) if latest.get('theta') else None,
                'rho': latest.get('rho'),
                'last_ts': latest.get('ts')
            })
    df_live = pd.DataFrame(rows)
    st.dataframe(df_live)

# Charts: show one chart per greek (delta, gamma, vega, theta) for the selected instruments
st.subheader("Real-time Greeks charts (live)")
chart_container = st.empty()

def plot_greeks(greek_key='delta'):
    plt.clf()
    fig, ax = plt.subplots(figsize=(10,5))
    with DATA_LOCK:
        rows = []
        for ik in DATA['selected_instruments']:
            for rec in DATA['series'].get(ik, []):
                rows.append({'ts': rec['ts'], 'instrument': ik, greek_key: rec.get(greek_key)})
    if not rows:
        st.info("No data yet for charts. Wait for polling to start at 09:20 IST (or force selection/polling).")
        return
    dff = pd.DataFrame(rows)
    dff['ts_dt'] = pd.to_datetime(dff['ts'])
    pivot = dff.pivot_table(index='ts_dt', columns='instrument', values=greek_key)
    pivot = pivot.fillna(method='ffill').fillna(method='bfill')
    pivot.plot(ax=ax, linewidth=1)
    ax.set_title(f"Live {greek_key} (selected instruments)")
    ax.set_xlabel("Time")
    ax.set_ylabel(greek_key)
    ax.grid(True)
    st.pyplot(fig)

# show 4 charts in tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Delta","Gamma","Vega","Theta (per day)","IV"]) 
with tab1:
    plot_greeks('delta')
with tab2:
    plot_greeks('gamma')
with tab3:
    plot_greeks('vega')
with tab4:
    plt.clf()
    fig, ax = plt.subplots(figsize=(10,5))
    with DATA_LOCK:
        rows = []
        for ik in DATA['selected_instruments']:
            for rec in DATA['series'].get(ik, []):
                theta_pd = (rec.get('theta') / 365) if rec.get('theta') else None
                rows.append({'ts': rec['ts'], 'instrument': ik, 'theta_pd': theta_pd})
    if not rows:
        st.info("No data yet for charts. Wait for polling to start at 09:20 IST (or force selection/polling).")
    else:
        dff = pd.DataFrame(rows)
        dff['ts_dt'] = pd.to_datetime(dff['ts'])
        pivot = dff.pivot_table(index='ts_dt', columns='instrument', values='theta_pd')
        pivot = pivot.fillna(method='ffill').fillna(method='bfill')
        pivot.plot(ax=ax, linewidth=1)
        ax.set_title("Live Theta (per day)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Theta (per day)")
        ax.grid(True)
        st.pyplot(fig)
with tab5:
    plot_greeks('iv')

st.markdown("---")
st.markdown("**Notes & tips**: \n"
            "- Provide your Upstox access token (paste in the top box) each day. Tokens expire at 03:30 IST.\n"
            "- The script selects strikes at 09:16 IST and polls between 09:20 - 15:20 IST. You can Force selection for testing.\n"
            "- This implementation polls the Upstox option-greek REST endpoint (v3) once per second for the selected instruments (keeps within the 50-instrument limit).\n"
            "- The greeks returned by Upstox are used (if present). If any greek is missing, the app will compute Black-Scholes greeks locally using the provided IV (code uses standard formulas in the app source).\n")

# End of file
