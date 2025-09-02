import os
import sys
import subprocess
import time
import threading
import json
from datetime import datetime, time as dtime
import pytz
import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from google.protobuf.json_format import MessageToDict

# ------- Protobuf Setup -------
# Ensure feeds_proto directory is in path
PROTO_DIR = os.path.join(os.getcwd(), "feeds_proto")
if PROTO_DIR not in sys.path:
    sys.path.append(PROTO_DIR)

PROTO_FILE = os.path.join(PROTO_DIR, "MarketDataFeedV3.proto")
PB2_FILE = os.path.join(PROTO_DIR, "MarketDataFeedV3_pb2.py")
PB2_GRPC = os.path.join(PROTO_DIR, "MarketDataFeedV3_pb2_grpc.py")

# Auto-generate pb2 files at runtime if missing
if not os.path.exists(PB2_FILE):
    try:
        subprocess.run([
            sys.executable, "-m", "grpc_tools.protoc",
            f"-I{PROTO_DIR}",
            f"--python_out={PROTO_DIR}",
            f"--grpc_python_out={PROTO_DIR}",
            PROTO_FILE
        ], check=True)
        st.info("Generated protobuf modules successfully.")
    except Exception as e:
        st.error(f"Failed to generate proto modules: {e}")
        st.stop()

# Import generated protobuf module
try:
    import MarketDataFeedV3_pb2 as pb
except Exception as e:
    st.error(f"Could not import MarketDataFeedV3_pb2: {e}")
    st.stop()

# ------- Config -------
IST = pytz.timezone("Asia/Kolkata")
API_V3_BASE = "https://api.upstox.com/v3"
DAILY_SELECTION = dtime(9, 16)
POLL_START = dtime(9, 20)
POLL_END = dtime(15, 20)
ITM_COUNT = 5
OTM_COUNT = 5

# Shared data
DATA = {
    "token": None,
    "selected_date": None,
    "instrument_keys": [],
    "series": defaultdict(list),
    "ws_url": None,
}

DATA_LOCK = threading.Lock()

# ------- Helpers -------
def ist_now():
    return datetime.now(IST)

def headers(token):
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}

def authorize_ws_url(token):
    url = f"{API_V3_BASE}/feed/market-data-feed/authorize"
    r = requests.get(url, headers=headers(token), timeout=15)
    r.raise_for_status()
    return r.json().get("data", {}).get("authorized_redirect_uri")

def fetch_option_contracts(token, underlying_key="NSE_INDEX|Nifty 50"):
    url = f"{API_V3_BASE}/option/contract"
    r = requests.get(url, headers=headers(token), params={"instrument_key": underlying_key}, timeout=15)
    r.raise_for_status()
    return r.json().get("data", [])

def fetch_ltp(token, keys):
    if not keys:
        return {}
    url = f"{API_V3_BASE}/market-quote/ltp"
    r = requests.get(url, headers=headers(token), params={"instrument_key": ",".join(keys)}, timeout=15)
    r.raise_for_status()
    out = {}
    for rec in r.json().get("data", []) or []:
        ik = rec.get("instrument_key") or rec.get("instrumentKey")
        price = rec.get("ltp") or rec.get("last_price")
        out[ik] = price
    return out

def choose_today_instruments(token):
    data = fetch_option_contracts(token)
    if not data:
        raise RuntimeError("No option contracts found.")
    df = pd.DataFrame(data)
    today = ist_now().date()
    df['expiry_dt'] = pd.to_datetime(df['expiry']).dt.date
    future = sorted([d for d in df['expiry_dt'].unique() if d >= today])
    expiry = future[0].isoformat() if future else df['expiry_dt'].min().isoformat()
    df = df[df['expiry_dt'] == pd.to_datetime(expiry)]
    spot = fetch_ltp(token, [df['instrument_key'].iloc[0]]) or 0
    spot_price = spot[next(iter(spot), "")] or (df['strike_price'].median())
    def pick(strikes, is_ce=True):
        strikes = sorted(set(strikes))
        atm = min(strikes, key=lambda x: abs(x - spot_price))
        idx = strikes.index(atm)
        if is_ce:
            lower = strikes[max(0, idx-ITM_COUNT):idx]
            higher = strikes[idx+1:idx+1+OTM_COUNT]
            return list(reversed(lower)) + higher
        else:
            higher = strikes[idx:idx+OTM_COUNT]
            lower = strikes[max(0, idx-ITM_COUNT):idx]
            return list(reversed(higher)) + list(reversed(lower))
    ce_str = pick(df[df['instrument_type']=="CE"]['strike_price'], is_ce=True)
    pe_str = pick(df[df['instrument_type']=="PE"]['strike_price'], is_ce=False)
    keys = []
    for s in ce_str:
        row = df[(df['strike_price']==s)&(df['instrument_type']=="CE")].iloc[0]
        keys.append(row['instrument_key'])
    for s in pe_str:
        row = df[(df['strike_price']==s)&(df['instrument_type']=="PE")].iloc[0]
        keys.append(row['instrument_key'])
    return keys, expiry

# ------- WebSocket Handler -------
def parse_proto_message(raw):
    try:
        resp = pb.FeedResponse()
        resp.ParseFromString(raw)
        return MessageToDict(resp, preserving_proto_field_name=True)
    except Exception:
        return None

class WSClient:
    def __init__(self, url, token, on_msg):
        self.url = url
        self.token = token
        self.on_msg = on_msg
        self.ws = None
        self.thread = None
    def _on_message(self, ws, msg):
        parsed = parse_proto_message(msg)
        if parsed:
            self.on_msg(parsed)
    def start(self):
        self.ws = websocket.WebSocketApp(
            self.url,
            header=[f"Authorization: Bearer {self.token}"],
            on_message=self._on_message)
        self.thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        self.thread.start()
    def subscribe(self, keys):
        if not self.ws:
            return
        msg = {
            "guid": str(int(time.time())),
            "method": "sub",
            "data": {"instrumentKeys": keys, "mode": "option_greeks"}
        }
        self.ws.send(json.dumps(msg))
    def stop(self):
        if self.ws:
            self.ws.close()

# ------- Background Thread -------
class Background(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.ws_client = None
    def run(self):
        while True:
            token = DATA.get("token")
            now = ist_now()
            if token and now.time() >= DAILY_SELECTION and DATA.get("selected_date") != now.date().isoformat():
                try:
                    keys, expiry = choose_today_instruments(token)
                    with DATA_LOCK:
                        DATA["selected_date"] = now.date().isoformat()
                        DATA["instrument_keys"] = keys
                        DATA["expiry"] = expiry
                        DATA["series"] = defaultdict(list)
                except Exception as e:
                    print("Selection failed:", e)
            if token and DATA.get("instrument_keys") and POLL_START <= now.time() <= POLL_END:
                if not DATA.get("ws_url"):
                    try:
                        DATA["ws_url"] = authorize_ws_url(token)
                    except Exception as e:
                        print("Authorize WS failed:", e)
                        time.sleep(5); continue
                if not self.ws_client:
                    self.ws_client = WSClient(DATA["ws_url"], token, self.on_data)
                    self.ws_client.start()
                    time.sleep(1)
                    self.ws_client.subscribe(DATA["instrument_keys"])
                time.sleep(POLL_INTERVAL_SECONDS)
            else:
                time.sleep(5)
    def on_data(self, parsed):
        ts = ist_now().strftime("%Y-%m-%d %H:%M:%S")
        found = parsed.get("feeds", {}) or parsed.get("data", {}) or parsed
        for item in (found.values() if isinstance(found, dict) else found):
            ik = item.get("instrument_key") or item.get("instrumentKey")
            if not ik: continue
            rec = {
                "ts": ts,
                "ltp": item.get("ltp"),
                "iv": item.get("implied_volatility"),
                "delta": item.get("delta"),
                "gamma": item.get("gamma"),
                "vega": item.get("vega"),
                "theta": item.get("theta"),
                "rho": item.get("rho"),
            }
            with DATA_LOCK:
                DATA["series"][ik].append(rec)

# ------- UI & Run -------
st.set_page_config(layout="wide")
st_autoref = st.experimental_set_query_params  # dummy to force refresh
st.title("Real-time NIFTY Option Greeks (WebSocket + Protobuf)")

token_input = st.text_input("Paste Upstox ACCESS TOKEN (daily)", type="password")
if st.button("Save token"):
    if token_input:
        DATA["token"] = token_input.strip()
        st.success("Saved token in memory")
    else:
        st.error("Empty token!")

st.metric("Selected Date", DATA.get("selected_date") or "-")
st.metric("Expiry", DATA.get("expiry") or "-")

if "bg" not in st.session_state:
    st.session_state.bg = Background()
    st.session_state.bg.start()

with DATA_LOCK:
    keys = DATA.get("instrument_keys")
    df_live = []
    for ik in keys:
        hist = DATA["series"].get(ik, [])
        latest = hist[-1] if hist else {}
        df_live.append({
            "instrument_key": ik,
            "ltp": latest.get("ltp"),
            "delta": latest.get("delta"),
            "gamma": latest.get("gamma"),
            "theta": latest.get("theta"),
            "vega": latest.get("vega"),
            "iv": latest.get("iv"),
            "timestamp": latest.get("ts"),
        })
df = pd.DataFrame(df_live)

st.subheader("Latest Snapshot")
st.dataframe(df)

st.subheader("Live Greeks Charts")
tabs = st.tabs(["Delta", "Gamma", "Vega", "Theta"])
for name, tab in zip(["delta","gamma","vega","theta"], tabs):
    with tab:
        plt.clf()
        fig, ax = plt.subplots(figsize=(10,4))
        with DATA_LOCK:
            recs = []
            for ik in keys:
                for rec in DATA["series"].get(ik, []):
                    recs.append({"ts": rec["ts"], "instrument": ik, name: rec.get(name)})
        if not recs:
            st.info("No data yet...")
        else:
            dfp = pd.DataFrame(recs)
            dfp["ts"] = pd.to_datetime(dfp["ts"])
            pivot = dfp.pivot(index="ts", columns="instrument", values=name).ffill().bfill()
            pivot.plot(ax=ax)
            ax.set_title(f"{name.capitalize()} over time")
            ax.set_xlabel("Time (IST)")
            ax.set_ylabel(name)
            st.pyplot(fig)
