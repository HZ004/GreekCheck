# app.py — Upstox NIFTY Greeks (Render-ready, auto-select on startup)
# - Select 5 ITM + 5 OTM per side at 09:16 IST; stream 09:20–15:20 IST
# - ALSO auto-select immediately if app starts mid-market and nothing is selected
# - Prefers WebSocket v3 + protobuf ('option_greeks'); falls back to REST polling
# - Generates/loads protobuf at runtime if feeds_proto/MarketDataFeedV3.proto provided
# - Shows live snapshot + Delta/Gamma/Vega/Theta charts
# ---------------------------------------------------------------

import os, sys, time, json, threading, math
from datetime import datetime, time as dtime
from collections import defaultdict, deque

import streamlit as st
import requests
import pandas as pd
import numpy as np
import pytz
from scipy.stats import norm
from scipy.optimize import brentq
import websocket
from google.protobuf.json_format import MessageToDict

# -------------------- CONFIG --------------------
IST = pytz.timezone("Asia/Kolkata")

API_V2_BASE = "https://api.upstox.com/v2"  # for option contract/chain (more reliable)
API_V3_BASE = "https://api.upstox.com/v3"  # for LTP/greeks + feed authorize

UNDERLYING_DEFAULT = "NSE_INDEX|Nifty 50"   # change if you want BankNifty etc.

# Trading-day automation
SELECT_TIME = dtime(9, 16)  # auto strike selection time
START_TIME  = dtime(9, 20)  # start streaming
STOP_TIME   = dtime(15, 20) # stop streaming

ITM_COUNT = 5
OTM_COUNT = 5
POLL_INTERVAL = 1.0         # seconds for REST fallback polling
RISK_FREE = 0.05            # used only in rare BS fallback calcs

# Protobuf locations (optional for WS)
PROTO_DIR = os.path.join(os.getcwd(), "feeds_proto")
PROTO_FILE = os.path.join(PROTO_DIR, "MarketDataFeedV3.proto")
PB2_PY = os.path.join(PROTO_DIR, "MarketDataFeedV3_pb2.py")

# -------------------- SHARED STATE --------------------
DATA_LOCK = threading.Lock()
DATA = {
    "token": None,
    "underlying_key": UNDERLYING_DEFAULT,

    "expiry_choices": [],      # yyyy-mm-dd strings
    "selected_expiry": None,   # user-chosen expiry (optional)

    "selected_date": None,     # yyyy-mm-dd (when selection done)
    "expiry_chosen": None,     # yyyy-mm-dd (the chosen expiry used for the day)
    "instrument_keys": [],     # final locked list for the day

    "series": defaultdict(list),  # instrument_key -> [{ts, ltp, iv, delta,...}, ...]

    "ws_url": None,            # from /v3/feed/market-data-feed/authorize
    "proto_available": False,  # pb2 loaded
    "pb_module": None,         # imported pb module
}

LOGS = deque(maxlen=600)
def log(msg):
    ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    s = f"[{ts}] {msg}"
    LOGS.append(s)
    print(s)

def now_ist():
    return datetime.now(IST)

def in_window(start_t, end_t, t=None):
    t = t or now_ist().time()
    return start_t <= t <= end_t

def headers(token):
    return {"Accept": "application/json", "Authorization": f"Bearer {token}"}

# -------------------- Black-Scholes fallback (rare) --------------------
def bs_price(opt_type, S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(0.0, S-K) if opt_type=="CE" else max(0.0, K-S)
    d1 = (math.log(S/K) + (r+0.5*sigma*sigma)*T)/(sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    if opt_type=="CE":
        return S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
    else:
        return K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def bs_greeks(opt_type, S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return {"delta":0,"gamma":0,"vega":0,"theta":0,"rho":0}
    d1 = (math.log(S/K) + (r+0.5*sigma*sigma)*T)/(sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    pdf = norm.pdf(d1)
    if opt_type=="CE":
        delta = norm.cdf(d1)
        theta = -(S*pdf*sigma)/(2*math.sqrt(T)) - r*K*math.exp(-r*T)*norm.cdf(d2)
        rho   = K*T*math.exp(-r*T)*norm.cdf(d2)
    else:
        delta = -norm.cdf(-d1)
        theta = -(S*pdf*sigma)/(2*math.sqrt(T)) + r*K*math.exp(-r*T)*norm.cdf(-d2)
        rho   = -K*T*math.exp(-r*T)*norm.cdf(-d2)
    gamma = pdf/(S*sigma*math.sqrt(T))
    vega  = S*pdf*math.sqrt(T)
    return {"delta":float(delta), "gamma":float(gamma), "vega":float(vega),
            "theta":float(theta/365.0), "rho":float(rho)}

def implied_vol(opt_type, price, S, K, T, r):
    if not price or price <= 0: return None
    f = lambda sig: bs_price(opt_type, S, K, T, r, sig) - price
    try:
        return brentq(f, 1e-6, 5.0, maxiter=200, xtol=1e-6)
    except Exception:
        return None

# -------------------- Upstox REST --------------------
def get_option_contracts(token, underlying_key):
    url = f"{API_V2_BASE}/option/contract"
    r = requests.get(url, headers=headers(token), params={"instrument_key": underlying_key}, timeout=15)
    r.raise_for_status()
    return r.json().get("data", [])

def get_ltp_v3(token, instrument_keys):
    if not instrument_keys: return {}
    url = f"{API_V3_BASE}/market-quote/ltp"
    r = requests.get(url, headers=headers(token), params={"instrument_key": ",".join(instrument_keys)}, timeout=15)
    r.raise_for_status()
    out = {}
    data = r.json().get("data", {})
    if isinstance(data, list):
        for d in data:
            k = d.get("instrument_key") or d.get("instrumentKey")
            out[k] = d.get("ltp") or d.get("last_price")
    elif isinstance(data, dict):
        for k,v in data.items():
            out[k] = v.get("ltp") or v.get("last_price")
    return out

def get_option_greeks_v3(token, instrument_keys):
    # REST fallback — some accounts may get nulls
    if not instrument_keys: return {}
    url = f"{API_V3_BASE}/market-quote/option-greek"
    r = requests.get(url, headers=headers(token), params={"instrument_key": ",".join(instrument_keys)}, timeout=15)
    r.raise_for_status()
    return r.json().get("data", {})

def authorize_ws_url(token):
    url = f"{API_V3_BASE}/feed/market-data-feed/authorize"
    r = requests.get(url, headers=headers(token), timeout=15)
    r.raise_for_status()
    d = r.json().get("data", {}) or {}
    return d.get("authorized_redirect_uri") or d.get("authorized_url")

# -------------------- Protobuf load/generate (optional) --------------------
def ensure_pb2_loaded():
    # If pb2 exists, import it; else try to generate from .proto if present
    if os.path.exists(PB2_PY):
        try:
            if PROTO_DIR not in sys.path: sys.path.insert(0, PROTO_DIR)
            import MarketDataFeedV3_pb2 as pb
            DATA["pb_module"] = pb
            DATA["proto_available"] = True
            log("Imported existing MarketDataFeedV3_pb2.py successfully.")
            return True
        except Exception as e:
            log(f"Import pb2 failed: {e}")

    # Try to generate if proto exists
    if os.path.exists(PROTO_FILE):
        try:
            from grpc_tools import protoc
            log("Attempting to generate pb2 with grpc_tools.protoc ...")
            rc = protoc.main([
                "protoc",
                f"-I{PROTO_DIR}",
                f"--python_out={PROTO_DIR}",
                f"--grpc_python_out={PROTO_DIR}",
                PROTO_FILE
            ])
            if rc != 0:
                log(f"grpc_tools.protoc returned rc={rc}")
                return False
            if PROTO_DIR not in sys.path: sys.path.insert(0, PROTO_DIR)
            import MarketDataFeedV3_pb2 as pb
            DATA["pb_module"] = pb
            DATA["proto_available"] = True
            log("Generated and imported MarketDataFeedV3_pb2.py successfully.")
            return True
        except Exception as e:
            log(f"Proto generation failed: {e}")
    else:
        log("No proto file found (feeds_proto/MarketDataFeedV3.proto). WS protobuf mode may be limited.")

    return False

def parse_proto_message_bytes(pb_module, raw):
    # Try to parse raw bytes into any message type defined in pb_module (best-effort)
    from google.protobuf.message import Message
    for name, obj in pb_module.__dict__.items():
        if isinstance(obj, type):
            inst = obj()
            if hasattr(inst, "ParseFromString"):
                try:
                    inst.ParseFromString(raw)
                    return MessageToDict(inst, preserving_proto_field_name=True)
                except Exception:
                    continue
    return None

# -------------------- Selection --------------------
def choose_strikes_for_today(token):
    """Pick nearest expiry (or user-selected), lock 5 ITM + 5 OTM for CE & PE."""
    oc = get_option_contracts(token, DATA["underlying_key"])
    if not oc:
        raise RuntimeError("No option contracts returned.")
    df = pd.DataFrame(oc)
    df["expiry_dt"] = pd.to_datetime(df["expiry"]).dt.date

    # Build expiry choices (first 6 to keep it short)
    exps = sorted({d.isoformat() for d in df["expiry_dt"].unique()})
    with DATA_LOCK:
        DATA["expiry_choices"] = exps[:6]

    # Pick expiry: user choice or nearest >= today
    today = now_ist().date()
    if DATA.get("selected_expiry"):
        expiry = datetime.strptime(DATA["selected_expiry"], "%Y-%m-%d").date()
    else:
        future = sorted([d for d in df["expiry_dt"].unique() if d >= today])
        if not future:
            future = [min(df["expiry_dt"])]
        expiry = future[0]

    df_exp = df[df["expiry_dt"] == expiry]
    # Underlying spot
    spot_map = get_ltp_v3(token, [DATA["underlying_key"]])
    spot = spot_map.get(DATA["underlying_key"])
    if spot is None:
        # fallback: midpoint of strikes
        spot = float(df_exp["strike_price"].median())

    ce_strikes = sorted(df_exp[df_exp["instrument_type"]=="CE"]["strike_price"].unique())
    pe_strikes = sorted(df_exp[df_exp["instrument_type"]=="PE"]["strike_price"].unique())
    if not ce_strikes or not pe_strikes:
        raise RuntimeError("No CE/PE strikes for chosen expiry.")

    def pick(strikes):
        atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i]-spot))
        itm = strikes[max(0, atm_idx-ITM_COUNT):atm_idx]
        otm = strikes[atm_idx+1:atm_idx+1+OTM_COUNT]
        return list(reversed(itm)) + otm

    pick_ce = pick(ce_strikes)
    pick_pe = pick(pe_strikes)

    keys = []
    for s in pick_ce:
        row = df_exp[(df_exp["strike_price"]==s)&(df_exp["instrument_type"]=="CE")]
        if not row.empty: keys.append(row.iloc[0]["instrument_key"])
    for s in pick_pe:
        row = df_exp[(df_exp["strike_price"]==s)&(df_exp["instrument_type"]=="PE")]
        if not row.empty: keys.append(row.iloc[0]["instrument_key"])

    return {
        "expiry": expiry.isoformat(),
        "spot": float(spot),
        "keys": keys,
        "ce_strikes": pick_ce,
        "pe_strikes": pick_pe
    }

# -------------------- WebSocket client --------------------
class WSClient:
    def __init__(self, url, token, cb):
        self.url = url
        self.token = token
        self.cb = cb
        self.ws = None
        self.thread = None
        self.connected = False

    def _on_open(self, ws):
        self.connected = True
        log("WebSocket opened.")

    def _on_close(self, ws, code, reason):
        self.connected = False
        log(f"WebSocket closed: {code} {reason}")

    def _on_error(self, ws, err):
        log(f"WebSocket error: {err}")

    def _on_message(self, ws, msg):
        try:
            if isinstance(msg, bytes):
                if DATA["proto_available"] and DATA["pb_module"]:
                    d = parse_proto_message_bytes(DATA["pb_module"], msg)
                    if d: self.cb(d); return
                # try json
                try:
                    d = json.loads(msg.decode("utf-8"))
                    self.cb(d); return
                except Exception:
                    return
            else:
                d = json.loads(msg)
                self.cb(d); return
        except Exception as e:
            log(f"WS message parse error: {e}")

    def start(self):
        headers = [f"Authorization: Bearer {self.token}"]
        self.ws = websocket.WebSocketApp(
            self.url, header=headers,
            on_open=self._on_open, on_close=self._on_close,
            on_error=self._on_error, on_message=self._on_message
        )
        self.thread = threading.Thread(target=self.ws.run_forever, kwargs={"ping_interval":20, "ping_timeout":5}, daemon=True)
        self.thread.start()

    def subscribe(self, keys, mode="option_greeks"):
        if not self.ws: return
        msg = {"guid": str(int(time.time())), "method": "sub", "data": {"instrumentKeys": keys, "mode": mode}}
        try:
            self.ws.send(json.dumps(msg))
            log(f"Subscribed {len(keys)} instruments (mode={mode}).")
        except Exception as e:
            log(f"WS subscribe failed: {e}")

    def stop(self):
        try:
            if self.ws: self.ws.close()
        except Exception: pass

# -------------------- Background thread --------------------
class Fetcher(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.ws = None
        self.stop_flag = False

    def stop(self):
        self.stop_flag = True
        if self.ws: self.ws.stop()

    def run(self):
        log("Background fetcher started.")
        while not self.stop_flag:
            token = DATA.get("token")
            if not token:
                time.sleep(0.5)
                continue

            # refresh expiry choices lazily
            if not DATA.get("expiry_choices"):
                try:
                    oc = get_option_contracts(token, DATA["underlying_key"])
                    if oc:
                        df = pd.DataFrame(oc)
                        df["expiry_dt"] = pd.to_datetime(df["expiry"]).dt.date
                        exps = sorted({d.isoformat() for d in df["expiry_dt"].unique()})
                        with DATA_LOCK:
                            DATA["expiry_choices"] = exps[:6]
                except Exception as e:
                    log(f"Fetch expiries failed: {e}")

            now = now_ist()
            today = now.date().isoformat()

            # AUTO-SELECTION:
            # 1) Standard rule: after 09:16 if not yet selected today
            # 2) EXTRA: If market already open and nothing selected, select immediately
            should_select = False
            with DATA_LOCK:
                nothing_selected = not DATA["instrument_keys"]
                not_selected_today = DATA.get("selected_date") != today

            if not_selected_today and now.time() >= SELECT_TIME:
                should_select = True
            elif not_selected_today and nothing_selected and in_window(dtime(9,15), dtime(15,30), now.time()):
                # app started mid-market; select right away
                should_select = True

            if should_select:
                try:
                    res = choose_strikes_for_today(token)
                    with DATA_LOCK:
                        DATA["selected_date"] = today
                        DATA["expiry_chosen"] = res["expiry"]
                        DATA["instrument_keys"] = res["keys"]
                        DATA["series"] = defaultdict(list)
                    log(f"Selected {len(res['keys'])} instruments for expiry {res['expiry']}.")
                except Exception as e:
                    log(f"Selection error: {e}")

            # STREAMING: only between 09:20–15:20 and if we have contracts
            with DATA_LOCK:
                ready = DATA["instrument_keys"] and in_window(START_TIME, STOP_TIME, now.time())

            if ready:
                # ensure ws url
                if not DATA.get("ws_url"):
                    try:
                        wsurl = authorize_ws_url(token)
                        if wsurl:
                            with DATA_LOCK: DATA["ws_url"] = wsurl
                            log("Acquired ws URL.")
                    except Exception as e:
                        log(f"Authorize WS failed: {e}")

                # prefer WS+proto when possible
                if DATA.get("ws_url") and DATA.get("proto_available"):
                    if not self.ws:
                        try:
                            self.ws = WSClient(DATA["ws_url"], token, self.on_parsed)
                            self.ws.start()
                            time.sleep(1)
                            self.ws.subscribe(DATA["instrument_keys"], mode="option_greeks")
                            log("WS client started & subscribed.")
                        except Exception as e:
                            log(f"WS start failed: {e}")
                            self.ws = None
                    time.sleep(POLL_INTERVAL)
                else:
                    # REST fallback polling
                    try:
                        g = get_option_greeks_v3(token, DATA["instrument_keys"])
                        ts = now.strftime("%Y-%m-%d %H:%M:%S")
                        entries = []
                        if isinstance(g, dict):
                            # may be dict keyed by instrument_key
                            for k, v in g.items():
                                if isinstance(v, dict):
                                    entries.append((k, v))
                        elif isinstance(g, list):
                            for v in g:
                                k = v.get("instrument_key") or v.get("instrumentKey")
                                if k: entries.append((k, v))
                        with DATA_LOCK:
                            for ik, v in entries:
                                rec = {
                                    "ts": ts,
                                    "ltp": v.get("ltp") or v.get("last_price"),
                                    "iv": v.get("iv") or v.get("implied_volatility"),
                                    "delta": v.get("delta"),
                                    "gamma": v.get("gamma"),
                                    "vega": v.get("vega"),
                                    "theta": v.get("theta"),
                                    "rho": v.get("rho"),
                                }
                                DATA["series"][ik].append(rec)
                    except Exception as e:
                        log(f"REST polling error: {e}")
                    time.sleep(POLL_INTERVAL)
            else:
                time.sleep(1)

    def on_parsed(self, parsed_dict):
        # Generic walker to extract instrument entries
        ts = now_ist().strftime("%Y-%m-%d %H:%M:%S")

        def walk(x, out):
            if isinstance(x, dict):
                if "instrument_key" in x or "instrumentKey" in x:
                    out.append(x)
                else:
                    for v in x.values(): walk(v, out)
            elif isinstance(x, list):
                for i in x: walk(i, out)

        items = []
        walk(parsed_dict, items)
        if not items: return
        with DATA_LOCK:
            for it in items:
                ik = it.get("instrument_key") or it.get("instrumentKey")
                if not ik: continue
                rec = {
                    "ts": ts,
                    "ltp": it.get("ltp") or it.get("last_price"),
                    "iv": it.get("implied_volatility") or it.get("iv"),
                    "delta": it.get("delta"),
                    "gamma": it.get("gamma"),
                    "vega": it.get("vega"),
                    "theta": it.get("theta"),
                    "rho": it.get("rho"),
                }
                DATA["series"][ik].append(rec)

# -------------------- Bootstrap proto (best-effort) --------------------
try:
    if ensure_pb2_loaded():
        log("Proto module available for websocket parsing.")
    else:
        log("Proto module not available; will use REST fallback if needed.")
except Exception as e:
    log(f"Proto init error: {e}")

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="NIFTY Greeks — Live (Upstox)", layout="wide")
st.title("NIFTY Options Greeks — Live (Upstox)")

# Token
col_t1, col_t2 = st.columns([3,1])
with col_t1:
    tok = st.text_input("Paste Upstox ACCESS TOKEN (daily)", value=DATA.get("token") or "", type="password")
with col_t2:
    if st.button("Save token"):
        if tok and len(tok) > 10:
            with DATA_LOCK:
                DATA["token"] = tok.strip()
            st.success("Token saved for this session.")
        else:
            st.error("Please paste a valid token.")

# Advanced controls
with st.expander("Advanced"):
    ukey = st.text_input("Underlying instrument_key", value=DATA.get("underlying_key", UNDERLYING_DEFAULT))
    if ukey != DATA.get("underlying_key"):
        with DATA_LOCK: DATA["underlying_key"] = ukey

    # Expiry choices (if available)
    with DATA_LOCK:
        ex_choices = DATA.get("expiry_choices", [])
        selected_exp = DATA.get("selected_expiry")

    opt = st.selectbox("Expiry (Auto=nearest)", options=["Auto (nearest)"] + ex_choices, index=0)
    if opt == "Auto (nearest)":
        if selected_exp is not None:
            with DATA_LOCK: DATA["selected_expiry"] = None
    else:
        with DATA_LOCK: DATA["selected_expiry"] = opt

# Background thread
if "fetcher" not in st.session_state:
    st.session_state.fetcher = Fetcher()
    st.session_state.fetcher.start()
    time.sleep(0.2)

# Manual buttons
col_b1, col_b2, col_b3 = st.columns(3)
with col_b1:
    if st.button("Force selection now"):
        if not DATA.get("token"):
            st.error("Save token first.")
        else:
            try:
                res = choose_strikes_for_today(DATA["token"])
                with DATA_LOCK:
                    DATA["selected_date"] = now_ist().date().isoformat()
                    DATA["expiry_chosen"] = res["expiry"]
                    DATA["instrument_keys"] = res["keys"]
                    DATA["series"] = defaultdict(list)
                st.success(f"Selected {len(res['keys'])} instruments for {res['expiry']}.")
            except Exception as e:
                st.error(f"Force selection failed: {e}")
with col_b2:
    if st.button("Resubscribe WS"):
        with DATA_LOCK:
            DATA["ws_url"] = None   # force fresh authorize
        st.info("Will re-authorize and resubscribe within a second.")
with col_b3:
    if st.button("Stop background"):
        if "fetcher" in st.session_state and st.session_state.fetcher:
            st.session_state.fetcher.stop()
            st.warning("Background fetcher stopped.")

# Logs
st.subheader("Logs")
st.code("\n".join(list(LOGS)[-60:]) or "No logs yet.")

# Selection summary
with DATA_LOCK:
    sel_date = DATA.get("selected_date") or "-"
    exp_used = DATA.get("expiry_chosen") or "-"
    keys = list(DATA.get("instrument_keys") or [])

st.subheader("Today's selection & latest values")
c1, c2, c3 = st.columns(3)
c1.metric("Selected Date", sel_date)
c2.metric("Expiry", exp_used)
c3.metric("Contracts", len(keys))

# Snapshot table
rows = []
with DATA_LOCK:
    for ik in keys:
        hist = DATA["series"].get(ik, [])
        last = hist[-1] if hist else {}
        rows.append({
            "instrument_key": ik,
            "ltp": last.get("ltp"),
            "iv": last.get("iv"),
            "delta": last.get("delta"),
            "gamma": last.get("gamma"),
            "vega": last.get("vega"),
            "theta": (last.get("theta")/365.0) if last.get("theta") else None,
            "rho": last.get("rho"),
            "last_ts": last.get("ts"),
        })
df_snap = pd.DataFrame(rows)
st.dataframe(df_snap)

# Charts
st.subheader("Live Greeks")
tabs = st.tabs(["Delta", "Gamma", "Vega", "Theta (per day)"])

def plot_series(greek_name, tab):
    with tab:
        recs = []
        with DATA_LOCK:
            for ik in keys:
                for r in DATA["series"].get(ik, []):
                    val = r.get(greek_name)
                    if greek_name == "theta" and val is not None:
                        val = val/365.0
                    recs.append({"ts": r["ts"], "instrument": ik, greek_name: val})
        if not recs:
            st.info("Waiting for data...")
        else:
            df = pd.DataFrame(recs)
            df["ts"] = pd.to_datetime(df["ts"])
            pv = df.pivot_table(index="ts", columns="instrument", values=greek_name).ffill().bfill()
            st.line_chart(pv)

plot_series("delta", tabs[0])
plot_series("gamma", tabs[1])
plot_series("vega",  tabs[2])
plot_series("theta", tabs[3])

st.markdown("---")
st.caption("Tip: If nothing appears, check Logs. Ensure your token has marketdata.read scope. "
           "For fastest updates use WS + protobuf (put MarketDataFeedV3.proto into feeds_proto/).")
