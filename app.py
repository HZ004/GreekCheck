# app.py
# Complete unified Streamlit app for real-time NIFTY option greeks (Upstox)
# - attempts protobuf generation at runtime (grpc_tools)
# - tries WebSocket V3 + protobuf feed for 'option_greeks'
# - falls back to REST polling of v3 option-greek endpoint if websocket/proto unavailable
# - auto-selects 5 ITM + 5 OTM strikes at 09:16 IST and streams from 09:20 - 15:20 IST
# - computes Black-Scholes Greeks locally as fallback
# NOTE: Put MarketDataFeedV3.proto (from Upstox) in feeds_proto/ if you want websocket mode.

import os, sys, time, json, traceback, threading
from datetime import datetime, date, time as dtime, timedelta
from collections import defaultdict, deque
import math

import requests
import streamlit as st
import pandas as pd
import numpy as np
import pytz
from scipy.stats import norm
from scipy.optimize import brentq
import websocket
from google.protobuf.json_format import MessageToDict

# -------------------- CONFIG --------------------
IST = pytz.timezone("Asia/Kolkata")
API_V2_BASE = "https://api.upstox.com/v2"
API_V3_BASE = "https://api.upstox.com/v3"
UNDERLYING_DEFAULT = "NSE_INDEX|Nifty 50"   # change if you use BANKNIFTY etc.
SELECT_TIME = dtime(9, 16)   # strike selection time IST
START_TIME = dtime(9, 20)    # start streaming
STOP_TIME  = dtime(15, 20)   # stop streaming
ITM_COUNT = 5
OTM_COUNT = 5
POLL_INTERVAL = 1.0   # seconds
RISK_FREE = 0.05

# Where proto is expected (if using WebSocket/protobuf parsing)
PROTO_DIR = os.path.join(os.getcwd(), "feeds_proto")
PROTO_FILE = os.path.join(PROTO_DIR, "MarketDataFeedV3.proto")
PB2_PY = os.path.join(PROTO_DIR, "MarketDataFeedV3_pb2.py")
PB2_GRPC_PY = os.path.join(PROTO_DIR, "MarketDataFeedV3_pb2_grpc.py")

# -------------------- SHARED STATE --------------------
DATA_LOCK = threading.Lock()
DATA = {
    "token": None,                    # user's Upstox access token (paste into UI daily)
    "underlying_key": UNDERLYING_DEFAULT,
    "expiry_choices": [],             # list of expiry strings (YYYY-MM-DD)
    "selected_expiry": None,
    "selected_date": None,            # date string 'YYYY-MM-DD' when selection done
    "instrument_keys": [],            # list of instrument_key strings chosen for the day
    "series": defaultdict(list),      # instrument_key -> list of records {ts, ltp, iv, delta,...}
    "ws_url": None,                   # authorized websocket url returned by authorize endpoint
    "proto_available": False,
    "pb_module": None,                # imported pb module if available
}

LOG_QUEUE = deque(maxlen=400)        # small ring buffer for logs shown in UI

def log(msg):
    ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    LOG_QUEUE.append(f"[{ts}] {msg}")
    print(LOG_QUEUE[-1])

# -------------------- UTILITIES --------------------
def now_ist():
    return datetime.now(IST)

def make_headers(token):
    return {"Accept": "application/json", "Authorization": f"Bearer {token}"}

# -------------------- BLACK-SCHOLES & IV --------------------
def bs_price(option_type, S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if option_type == "CE" else max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "CE":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_greeks(option_type, S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    pdf = norm.pdf(d1)
    if option_type == "CE":
        delta = norm.cdf(d1)
        theta = - (S * pdf * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r*T) * norm.cdf(d2)
        rho = K * T * math.exp(-r*T) * norm.cdf(d2)
    else:
        delta = -norm.cdf(-d1)
        theta = - (S * pdf * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r*T) * norm.cdf(-d2)
        rho = -K * T * math.exp(-r*T) * norm.cdf(-d2)
    gamma = pdf / (S * sigma * math.sqrt(T))
    vega = S * pdf * math.sqrt(T)
    return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega), "theta": float(theta/365.0), "rho": float(rho)}

def implied_vol(option_type, market_price, S, K, T, r):
    if market_price is None or market_price <= 0:
        return None
    def f(sigma):
        return bs_price(option_type, S, K, T, r, sigma) - market_price
    try:
        # reasonable bracket
        return brentq(f, 1e-6, 5.0, maxiter=200, xtol=1e-6)
    except Exception:
        return None

# -------------------- Upstox REST wrappers --------------------
def get_option_contracts(token, underlying_key):
    url = f"{API_V2_BASE}/option/contract"
    try:
        r = requests.get(url, headers=make_headers(token), params={"instrument_key": underlying_key}, timeout=15)
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception as e:
        log(f"get_option_contracts error: {e}")
        raise

def get_option_chain(token, underlying_key, expiry=None):
    url = f"{API_V2_BASE}/option/chain"
    params = {"instrument_key": underlying_key}
    if expiry:
        params["expiry_date"] = expiry
    r = requests.get(url, headers=make_headers(token), params=params, timeout=15)
    r.raise_for_status()
    return r.json().get("data", {})

def get_ltp_v3(token, instrument_keys):
    if not instrument_keys:
        return {}
    keys = ",".join(instrument_keys)
    url = f"{API_V3_BASE}/market-quote/ltp"
    r = requests.get(url, headers=make_headers(token), params={"instrument_key": keys}, timeout=15)
    r.raise_for_status()
    data = r.json().get("data", {})
    out = {}
    if isinstance(data, list):
        for item in data:
            k = item.get("instrument_key") or item.get("instrumentKey")
            if k:
                out[k] = item.get("ltp") or item.get("last_price")
    elif isinstance(data, dict):
        for k,v in data.items():
            out[k] = v.get("ltp") or v.get("last_price")
    return out

def get_option_greeks_v3(token, instrument_keys):
    # REST fallback - may return nulls for some accounts
    if not instrument_keys:
        return {}
    keys = ",".join(instrument_keys)
    url = f"{API_V3_BASE}/market-quote/option-greek"
    r = requests.get(url, headers=make_headers(token), params={"instrument_key": keys}, timeout=15)
    r.raise_for_status()
    return r.json().get("data", {})

def get_feed_authorize_url(token):
    url = f"{API_V3_BASE}/feed/market-data-feed/authorize"
    r = requests.get(url, headers=make_headers(token), timeout=15)
    r.raise_for_status()
    return r.json().get("data", {}).get("authorized_redirect_uri") or r.json().get("data", {}).get("authorized_url")

# -------------------- Protobuf generation/import (attempt) --------------------
def try_load_proto_module():
    # if pb2 already exists, import it
    if os.path.exists(PB2_PY):
        try:
            # Ensure Python can import from feeds_proto
            if PROTO_DIR not in sys.path:
                sys.path.insert(0, PROTO_DIR)
            import MarketDataFeedV3_pb2 as pb
            DATA["proto_available"] = True
            DATA["pb_module"] = pb
            log("Imported existing MarketDataFeedV3_pb2.py successfully.")
            return True
        except Exception as e:
            log(f"Import existing pb2 failed: {e}")
    # try to generate using grpc_tools.protoc if available
    try:
        from grpc_tools import protoc
        if not os.path.exists(PROTO_FILE):
            log("Proto file not found at feeds_proto/MarketDataFeedV3.proto; cannot generate pb2.")
            return False
        args = [
            'protoc',
            f'-I{PROTO_DIR}',
            f'--python_out={PROTO_DIR}',
            f'--grpc_python_out={PROTO_DIR}',
            PROTO_FILE
        ]
        log("Attempting to generate pb2 with grpc_tools.protoc ...")
        rc = protoc.main(args)
        if rc != 0:
            log(f"protoc returned non-zero rc: {rc}")
            return False
        # import now
        if PROTO_DIR not in sys.path:
            sys.path.insert(0, PROTO_DIR)
        import MarketDataFeedV3_pb2 as pb
        DATA["proto_available"] = True
        DATA["pb_module"] = pb
        log("Generated and imported MarketDataFeedV3_pb2.py successfully.")
        return True
    except Exception as e:
        log(f"Proto generation/import failed: {e}")
        return False

# -------------------- Protobuf parsing helper (brute-force attempt) --------------------
def parse_proto_bytes_try_all(pb_module, raw_bytes):
    # Try to parse raw_bytes into any message type defined in pb_module.
    from google.protobuf.message import Message
    for name, obj in pb_module.__dict__.items():
        try:
            if isinstance(obj, type):
                inst = obj()
                # only try parse if it has ParseFromString
                if hasattr(inst, "ParseFromString"):
                    try:
                        inst.ParseFromString(raw_bytes)
                        # got one that parsed without exception
                        d = MessageToDict(inst, preserving_proto_field_name=True)
                        return d
                    except Exception:
                        continue
        except Exception:
            continue
    return None

# -------------------- WebSocket client wrapper --------------------
class UpstoxWSClient:
    def __init__(self, url, token, on_parsed):
        self.url = url
        self.token = token
        self.on_parsed = on_parsed
        self.ws = None
        self.thread = None
        self.connected = False

    def _on_open(self, ws):
        log("WebSocket opened.")
        self.connected = True

    def _on_error(self, ws, err):
        log(f"WebSocket error: {err}")

    def _on_close(self, ws, code, reason):
        log(f"WebSocket closed: {code} {reason}")
        self.connected = False

    def _on_message(self, ws, message):
        # message may be bytes or str
        try:
            if isinstance(message, bytes):
                # try parse using pb_module if available
                if DATA.get("proto_available") and DATA.get("pb_module"):
                    parsed = parse_proto_bytes_try_all(DATA["pb_module"], message)
                    if parsed:
                        self.on_parsed(parsed)
                        return
                # fallback try decode as utf-8 JSON
                try:
                    txt = message.decode("utf-8")
                    j = json.loads(txt)
                    self.on_parsed(j)
                    return
                except Exception:
                    # cannot parse
                    return
            else:
                # text message
                try:
                    j = json.loads(message)
                    self.on_parsed(j)
                    return
                except Exception:
                    return
        except Exception as e:
            log(f"WS message handling error: {e}")

    def start(self):
        headers = [f"Authorization: Bearer {self.token}"]
        self.ws = websocket.WebSocketApp(self.url,
                                         header=headers,
                                         on_open=self._on_open,
                                         on_message=self._on_message,
                                         on_error=self._on_error,
                                         on_close=self._on_close)
        self.thread = threading.Thread(target=self.ws.run_forever, kwargs={"ping_interval":20, "ping_timeout":5}, daemon=True)
        self.thread.start()

    def send_subscribe(self, instrument_keys, mode="option_greeks"):
        # send subscribe JSON
        sub = {"guid": str(int(time.time())), "method": "sub", "data": {"instrumentKeys": instrument_keys, "mode": mode}}
        try:
            self.ws.send(json.dumps(sub))
            log(f"Sent subscribe msg for {len(instrument_keys)} instruments (mode={mode}).")
        except Exception as e:
            log(f"Failed to send subscribe: {e}")

    def stop(self):
        try:
            if self.ws:
                self.ws.close()
        except Exception:
            pass

# -------------------- Background fetcher --------------------
class BackgroundFetcher(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.ws_client = None
        self._stop = False

    def stop(self):
        self._stop = True
        if self.ws_client:
            self.ws_client.stop()

    def run(self):
        log("Background fetcher started.")
        while not self._stop:
            token = DATA.get("token")
            if not token:
                time.sleep(1)
                continue
            try:
                # refresh expiry choices if empty
                if not DATA.get("expiry_choices"):
                    try:
                        oc = get_option_contracts(token, DATA.get("underlying_key"))
                        if oc:
                            df = pd.DataFrame(oc)
                            # normalise expiry
                            df["expiry_dt"] = pd.to_datetime(df["expiry"]).dt.date
                            ex = sorted(list({d.isoformat() for d in df["expiry_dt"].unique()}))
                            with DATA_LOCK:
                                DATA["expiry_choices"] = ex[:6]
                                if not DATA.get("selected_expiry"):
                                    DATA["selected_expiry"] = ex[0] if ex else None
                            log(f"Found expiries: {ex[:6]}")
                    except Exception as e:
                        log(f"Could not fetch expiries: {e}")
                # 1) Do selection at SELECT_TIME if not already selected today
                now = now_ist()
                today_str = now.date().isoformat()
                if now.time() >= SELECT_TIME and DATA.get("selected_date") != today_str:
                    try:
                        out = choose_today(token)
                        with DATA_LOCK:
                            DATA["selected_date"] = today_str
                            DATA["instrument_keys"] = out["selected_instruments"]
                            DATA["expiry_chosen"] = out.get("expiry")
                            DATA["series"].clear()
                            DATA["series"] = defaultdict(list)
                        log(f"Selected {len(out['selected_instruments'])} instruments for expiry {out.get('expiry')}.")
                    except Exception as e:
                        log(f"Selection error: {e}")

                # 2) If within streaming window and instruments present -> attempt websocket else REST poll
                if DATA.get("instrument_keys") and START_TIME <= now.time() <= STOP_TIME:
                    # Try websocket if proto available and ws url available
                    if not DATA.get("proto_available"):
                        # try to load proto once
                        try_load_proto_module = try_load_proto_module()
                        if try_load_proto_module:
                            log("Proto module loaded; will prefer websocket feed if available.")
                    # ensure authorized ws url
                    if not DATA.get("ws_url"):
                        try:
                            url = get_feed_authorize_url(token)
                            if url:
                                with DATA_LOCK:
                                    DATA["ws_url"] = url
                                log(f"Acquired ws URL.")
                            else:
                                log("Authorize endpoint returned no ws url.")
                        except Exception as e:
                            log(f"Authorize ws url error: {e}")
                    # If proto available and ws_url present -> use websocket (event driven)
                    if DATA.get("proto_available") and DATA.get("ws_url"):
                        if not self.ws_client:
                            try:
                                self.ws_client = UpstoxWSClient(DATA["ws_url"], token, self.on_parsed_feed)
                                self.ws_client.start()
                                time.sleep(1)
                                self.ws_client.send_subscribe(DATA["instrument_keys"], mode="option_greeks")
                                log("Websocket client started & subscribed.")
                            except Exception as e:
                                log(f"Failed to start websocket client: {e}")
                                self.ws_client = None
                        # when websocket is active, messages will invoke on_parsed_feed
                        time.sleep(POLL_INTERVAL)
                    else:
                        # fallback: REST polling of option-greek endpoint every POLL_INTERVAL
                        try:
                            greeks = get_option_greeks_v3(token, DATA["instrument_keys"])
                            ts = now.strftime("%Y-%m-%d %H:%M:%S")
                            # greeks may be dict keyed by instrument_key or a list
                            entries = []
                            if isinstance(greeks, dict):
                                # sometimes greeks returned as dict or single object
                                if all(k.startswith("NSE") or "|" in k for k in greeks.keys()):
                                    for ik, val in greeks.items():
                                        entries.append((ik, val))
                                else:
                                    # Maybe list wrapped in dict
                                    for val in greeks.get("list", []) if greeks.get("list") else []:
                                        ik = val.get("instrument_key") or val.get("instrumentKey")
                                        entries.append((ik, val))
                            elif isinstance(greeks, list):
                                for val in greeks:
                                    ik = val.get("instrument_key") or val.get("instrumentKey")
                                    entries.append((ik, val))
                            else:
                                entries = []
                            with DATA_LOCK:
                                for ik, val in entries:
                                    if not ik:
                                        continue
                                    rec = {
                                        "ts": ts,
                                        "ltp": val.get("ltp") or val.get("last_price"),
                                        "iv": val.get("iv") or val.get("implied_volatility"),
                                        "delta": val.get("delta"),
                                        "gamma": val.get("gamma"),
                                        "vega": val.get("vega"),
                                        "theta": val.get("theta"),
                                        "rho": val.get("rho")
                                    }
                                    DATA["series"][ik].append(rec)
                        except Exception as e:
                            log(f"REST greeks polling error: {e}")
                        time.sleep(POLL_INTERVAL)
                else:
                    # outside streaming window
                    time.sleep(2)
            except Exception as e:
                log(f"Background exception: {e}")
                time.sleep(2)

    def on_parsed_feed(self, parsed):
        # parsed is dict from protobuf MessageToDict or JSON
        # try to find instrument entries
        ts = now_ist().strftime("%Y-%m-%d %H:%M:%S")
        def find_items(d):
            found = []
            if isinstance(d, dict):
                # direct instrument entries e.g., with keys instrument_key
                if "instrument_key" in d or "instrumentKey" in d:
                    found.append(d)
                else:
                    for v in d.values():
                        found.extend(find_items(v))
            elif isinstance(d, list):
                for item in d:
                    found.extend(find_items(item))
            return found
        try:
            items = find_items(parsed)
            with DATA_LOCK:
                for it in items:
                    ik = it.get("instrument_key") or it.get("instrumentKey")
                    if not ik:
                        continue
                    rec = {
                        "ts": ts,
                        "ltp": it.get("ltp") or it.get("last_price"),
                        "iv": it.get("implied_volatility") or it.get("iv"),
                        "delta": it.get("delta"),
                        "gamma": it.get("gamma"),
                        "vega": it.get("vega"),
                        "theta": it.get("theta"),
                        "rho": it.get("rho")
                    }
                    DATA["series"][ik].append(rec)
        except Exception as e:
            log(f"on_parsed_feed error: {e}")

# -------------------- Selection helper --------------------
def choose_today(token):
    # picks nearest expiry, picks ATM from underlying's ltp and selects 5 ITM + 5 OTM for CE and PE (one instrument_key each)
    oc = get_option_contracts(token, DATA.get("underlying_key"))
    if not oc:
        raise RuntimeError("No option contracts returned")
    df = pd.DataFrame(oc)
    df["expiry_dt"] = pd.to_datetime(df["expiry"]).dt.date
    today = now_ist().date()
    future_exps = sorted([d for d in df["expiry_dt"].unique() if d >= today])
    if not future_exps:
        raise RuntimeError("No upcoming expiries in contract list")
    expiry_choice = str(future_exps[0])
    df_exp = df[df["expiry_dt"] == pd.to_datetime(expiry_choice).date()]
    # get underlying LTP
    underlying_ltp_map = get_ltp_v3(token, [DATA.get("underlying_key")])
    underlying_price = underlying_ltp_map.get(DATA.get("underlying_key"))
    if underlying_price is None:
        # fallback to median strike
        underlying_price = float(df_exp["strike_price"].median())
    # pick strikes
    ce_strikes = sorted(df_exp[df_exp["instrument_type"] == "CE"]["strike_price"].unique())
    pe_strikes = sorted(df_exp[df_exp["instrument_type"] == "PE"]["strike_price"].unique())
    def pick_side(strikes):
        if not isinstance(strikes, (list, np.ndarray)) or len(strikes) == 0:
            return []
        atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - underlying_price))
        low = strikes[max(0, atm_idx - ITM_COUNT): atm_idx]
        high = strikes[atm_idx+1: atm_idx+1 + OTM_COUNT]
        return list(reversed(low)) + high
    selected_ce = pick_side(ce_strikes)
    selected_pe = pick_side(pe_strikes)
    selected_instruments = []
    # map strike->instrument_key (choose first matching)
    for s in selected_ce:
        row = df_exp[(df_exp["strike_price"] == s) & (df_exp["instrument_type"] == "CE")]
        if not row.empty:
            selected_instruments.append(row.iloc[0]["instrument_key"])
    for s in selected_pe:
        row = df_exp[(df_exp["strike_price"] == s) & (df_exp["instrument_type"] == "PE")]
        if not row.empty:
            selected_instruments.append(row.iloc[0]["instrument_key"])
    return {"underlying_price": underlying_price, "selected_instruments": selected_instruments, "expiry": expiry_choice, "ce_strikes": selected_ce, "pe_strikes": selected_pe}

# -------------------- Init proto/module (attempt) --------------------
# Attempt to load/generate pb2 module now (best-effort)
try:
    proto_ok = try_load_proto_module()
    if proto_ok:
        log("Proto module available for websocket parsing.")
    else:
        log("Proto module NOT available. Will fallback to REST polling if websocket cannot be used.")
except Exception as e:
    log(f"Proto load attempt raised: {e}")

# -------------------- Start background fetcher thread --------------------
if "bg_fetcher" not in st.session_state:
    st.session_state.bg_fetcher = BackgroundFetcher()
    st.session_state.bg_fetcher.start()
    time.sleep(0.2)

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="NIFTY Greeks (Upstox) — Live", layout="wide")
st.title("NIFTY Options Greeks — Live (Upstox)")

# token input
col1, col2 = st.columns([3,1])
with col1:
    token_in = st.text_input("Paste Upstox ACCESS_TOKEN (valid for the day)", value=DATA.get("token") or "", type="password")
with col2:
    if st.button("Save token"):
        if token_in and len(token_in) > 10:
            with DATA_LOCK:
                DATA["token"] = token_in.strip()
            st.success("Token saved in memory for this run.")
        else:
            st.error("Please paste a valid token string.")

# underlying / expiry controls
with st.expander("Advanced: underlying & expiry"):
    ukey = st.text_input("Underlying instrument_key", value=DATA.get("underlying_key", UNDERLYING_DEFAULT))
    if ukey != DATA.get("underlying_key"):
        with DATA_LOCK:
            DATA["underlying_key"] = ukey
    # show expiry choices
    with DATA_LOCK:
        ex_choices = DATA.get("expiry_choices", [])
    expiry_sel = st.selectbox("Expiry (nearest by default)", options=["Auto (nearest)"] + ex_choices, index=0 if ex_choices else 0)
    if expiry_sel != "Auto (nearest)":
        with DATA_LOCK:
            DATA["selected_expiry"] = expiry_sel

# Force selection for testing
if st.button("Force selection now (for testing)"):
    if not DATA.get("token"):
        st.error("Save a token first.")
    else:
        try:
            out = choose_today(DATA.get("token"))
            with DATA_LOCK:
                DATA["selected_date"] = now_ist().date().isoformat()
                DATA["instrument_keys"] = out["selected_instruments"]
                DATA["expiry_chosen"] = out["expiry"]
                DATA["series"] = defaultdict(list)
            st.success(f"Forced selection done: {len(out['selected_instruments'])} instruments.")
        except Exception as e:
            st.error(f"Force selection failed: {e}")

# Stop / Start background fetcher controls
colA, colB = st.columns(2)
with colA:
    if st.button("Start background fetcher"):
        if "bg_fetcher" not in st.session_state or not st.session_state.bg_fetcher.is_alive():
            st.session_state.bg_fetcher = BackgroundFetcher()
            st.session_state.bg_fetcher.start()
            st.success("Background fetcher started.")
        else:
            st.info("Background fetcher already running.")
with colB:
    if st.button("Stop background fetcher"):
        if "bg_fetcher" in st.session_state:
            st.session_state.bg_fetcher.stop()
            st.session_state.bg_fetcher = None
            st.warning("Background fetcher stopped.")

# Show logs (recent)
st.subheader("Logs")
log_text = "\n".join(list(LOG_QUEUE)[-50:])
st.code(log_text or "No logs yet.")

# Show selected & live snapshot
st.subheader("Today's selection & latest values")
with DATA_LOCK:
    selected = list(DATA.get("instrument_keys", []))
    expiry_chosen = DATA.get("expiry_chosen")
    selected_date = DATA.get("selected_date")
st.metric("Selected date", selected_date or "-")
st.metric("Expiry", expiry_chosen or "-")
st.metric("Total contracts", len(selected))

# Build a live dataframe snapshot
rows = []
with DATA_LOCK:
    for ik in selected:
        hist = DATA["series"].get(ik, [])
        latest = hist[-1] if hist else {}
        rows.append({
            "instrument_key": ik,
            "ltp": latest.get("ltp"),
            "iv": latest.get("iv"),
            "delta": latest.get("delta"),
            "gamma": latest.get("gamma"),
            "vega": latest.get("vega"),
            "theta_per_day": (latest.get("theta")/365.0) if latest.get("theta") else latest.get("theta"),
            "rho": latest.get("rho"),
            "last_ts": latest.get("ts")
        })
df_snapshot = pd.DataFrame(rows)
st.dataframe(df_snapshot)

# Charts for Greeks
st.subheader("Live Greeks charts")
tabs = st.tabs(["Delta","Gamma","Vega","Theta(per day)"])
with tabs[0]:
    if df_snapshot.empty:
        st.info("No data yet.")
    else:
        # For time-series plotting create pivoted df
        recs = []
        with DATA_LOCK:
            for ik in selected:
                for r in DATA["series"].get(ik, []):
                    recs.append({"ts": r["ts"], "instrument": ik, "delta": r.get("delta")})
        if not recs:
            st.info("Waiting for feed data...")
        else:
            dfp = pd.DataFrame(recs)
            dfp["ts_dt"] = pd.to_datetime(dfp["ts"])
            pivot = dfp.pivot_table(index="ts_dt", columns="instrument", values="delta").ffill().bfill()
            st.line_chart(pivot)
with tabs[1]:
    # gamma
    recs = []
    with DATA_LOCK:
        for ik in selected:
            for r in DATA["series"].get(ik, []):
                recs.append({"ts": r["ts"], "instrument": ik, "gamma": r.get("gamma")})
    if not recs:
        st.info("Waiting for feed data...")
    else:
        dfp = pd.DataFrame(recs); dfp["ts_dt"] = pd.to_datetime(dfp["ts"])
        st.line_chart(dfp.pivot_table(index="ts_dt", columns="instrument", values="gamma").ffill().bfill())
with tabs[2]:
    # vega
    recs = []
    with DATA_LOCK:
        for ik in selected:
            for r in DATA["series"].get(ik, []):
                recs.append({"ts": r["ts"], "instrument": ik, "vega": r.get("vega")})
    if not recs:
        st.info("Waiting for feed data...")
    else:
        dfp = pd.DataFrame(recs); dfp["ts_dt"] = pd.to_datetime(dfp["ts"])
        st.line_chart(dfp.pivot_table(index="ts_dt", columns="instrument", values="vega").ffill().bfill())
with tabs[3]:
    # theta per day (theta in feed is per year in many implementations)
    recs = []
    with DATA_LOCK:
        for ik in selected:
            for r in DATA["series"].get(ik, []):
                theta_pd = (r.get("theta")/365.0) if r.get("theta") else None
                recs.append({"ts": r["ts"], "instrument": ik, "theta": theta_pd})
    if not recs:
        st.info("Waiting for feed data...")
    else:
        dfp = pd.DataFrame(recs); dfp["ts_dt"] = pd.to_datetime(dfp["ts"])
        st.line_chart(dfp.pivot_table(index="ts_dt", columns="instrument", values="theta").ffill().bfill())

st.markdown("---")
st.markdown("**Notes:** If you do not see greeks updating: (1) check the logs above; (2) ensure your Upstox token has `marketdata.read` scope; (3) if REST greeks return null, enable websocket mode by placing `MarketDataFeedV3.proto` into `feeds_proto/` and redeploy (the app will try to auto-generate pb2 on startup if `grpc_tools` is present).")
