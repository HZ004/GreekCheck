# app.py
import streamlit as st
import requests
import websocket
import threading
import time
import json
from datetime import datetime, time as dtime, timedelta
import pytz
import pandas as pd
import numpy as np
import math
from collections import defaultdict
from google.protobuf.json_format import MessageToDict

# IMPORT the generated protobuf module you created from MarketDataFeedV3.proto
# Ensure the generated pb2 file is in feeds_proto/MarketDataFeedV3_pb2.py
try:
    from feeds_proto import MarketDataFeedV3_pb2 as pb
except Exception as e:
    st.exception("Could not import MarketDataFeedV3_pb2. Make sure you generated it with protoc and it's in feeds_proto/. Error: " + str(e))
    raise

# ====== CONFIG ======
IST = pytz.timezone("Asia/Kolkata")
API_BASE_V2 = "https://api.upstox.com/v2"
API_BASE_V3 = "https://api.upstox.com/v3"
UNDERLYING_KEY_DEFAULT = "NSE_INDEX|Nifty 50"   # change if needed
SELECT_TIME = dtime(9, 16)
START_TIME = dtime(9, 20)
STOP_TIME = dtime(15, 20)
POLL_INTERVAL_SECONDS = 1
ITM_COUNT = 5
OTM_COUNT = 5
RISK_FREE_RATE = 0.05

# ====== SHARED STATE (thread-safe) ======
DATA_LOCK = threading.Lock()
DATA = {
    "token": None,
    "selected_date": None,
    "underlying_key": UNDERLYING_KEY_DEFAULT,
    "expiry": None,
    "instrument_keys": [],           # list of strings (e.g. "NSE_FO|12345")
    "series": defaultdict(list),     # instrument_key -> list of {ts, ltp, delta, gamma, vega, theta, rho, iv}
    "feeder_url": None,              # authorized websocket url
}

# ====== Utils ======
def ist_now():
    return datetime.now(IST)

def make_headers(token):
    return {"Accept": "application/json", "Authorization": f"Bearer {token}"}

# ====== REST helpers ======
def fetch_option_contracts(token, underlying_key):
    """GET /v2/option/contract?instrument_key=UNDERLYING"""
    url = f"{API_BASE_V2}/option/contract"
    params = {"instrument_key": underlying_key}
    r = requests.get(url, headers=make_headers(token), params=params, timeout=15)
    r.raise_for_status()
    return r.json().get("data", [])

def fetch_option_chain(token, underlying_key, expiry=None):
    """GET /v2/option/chain (if v2 supports) OR use option contract + filter by expiry"""
    url = f"{API_BASE_V2}/option/chain"
    params = {"instrument_key": underlying_key}
    if expiry:
        params["expiry_date"] = expiry
    r = requests.get(url, headers=make_headers(token), params=params, timeout=15)
    r.raise_for_status()
    return r.json().get("data", {})

def fetch_ltp_map(token, instrument_keys):
    """v3 LTP endpoint accepts multiple instrument_key (max allowed by Upstox)"""
    if not instrument_keys:
        return {}
    keys = ",".join(instrument_keys)
    url = f"{API_BASE_V3}/market-quote/ltp"
    params = {"instrument_key": keys}
    r = requests.get(url, headers=make_headers(token), params=params, timeout=15)
    r.raise_for_status()
    data = r.json().get("data", {})
    ltp_map = {}
    if isinstance(data, list):
        for item in data:
            ik = item.get("instrument_key") or item.get("instrumentKey")
            if ik:
                ltp_map[ik] = item.get("ltp") or item.get("last_price") or None
    elif isinstance(data, dict):
        for k,v in data.items():
            ltp_map[k] = v.get("ltp") or v.get("last_price")
    return ltp_map

def fetch_option_greeks_rest(token, instrument_keys):
    """Fallback: get option greeks via REST v3 (may return null for some accounts)"""
    if not instrument_keys:
        return {}
    keys = ",".join(instrument_keys)
    url = f"{API_BASE_V3}/market-quote/option-greek"
    params = {"instrument_key": keys}
    r = requests.get(url, headers=make_headers(token), params=params, timeout=15)
    r.raise_for_status()
    return r.json().get("data", {})

def get_marketdata_authorized_url(token):
    """GET /v3/feed/market-data-feed/authorize -> returns an authorized websocket URL you should use"""
    url = f"{API_BASE_V3}/feed/market-data-feed/authorize"
    r = requests.get(url, headers=make_headers(token), timeout=15)
    r.raise_for_status()
    data = r.json().get("data", {})
    # field commonly named 'authorized_redirect_uri' or 'authorized_url' depending on docs
    return data.get("authorized_redirect_uri") or data.get("authorized_url") or data.get("authorizedWsUrl") or data.get("authorizedUrl")

# ====== Protobuf decode helper ======
def parse_feed_message(raw_bytes):
    """
    Parse the binary message payload using the generated pb module.
    The top-level message in Upstox proto can vary by name; community examples
    typically parse into a FeedResponse or similar message.
    We'll attempt common message types from the proto; if you generated pb from the official proto,
    inspect pb.__dict__ to find the proper message class if the below fails.
    """
    # Try likely top-level container names (may differ with Upstox proto version)
    for cls_name in ("FeedResponse","FeedResponseV3","Response","MarketFeed","Feed"):
        if hasattr(pb, cls_name):
            msg = getattr(pb, cls_name)()
            try:
                msg.ParseFromString(raw_bytes)
                return MessageToDict(msg, preserving_proto_field_name=True)
            except Exception:
                continue
    # If none matched, try the most common found in sample code: 'FeederResponse' or 'FeedEntity'
    # Last resort: try a generic parse into any top-level message defined in generated pb
    # Iterate all message types — try those with ParseFromString
    for name, obj in pb.__dict__.items():
        if isinstance(obj, type):
            try:
                instance = obj()
                instance.ParseFromString(raw_bytes)
                return MessageToDict(instance, preserving_proto_field_name=True)
            except Exception:
                continue
    # If still not parsed, return None
    return None

# ====== WebSocket handling ======
class UpstoxWS:
    def __init__(self, ws_url, token, on_data):
        """
        ws_url: authorized websocket url obtained via REST authorize endpoint
        token: access token (string)
        on_data: callback function parsed_dict -> None
        """
        self.ws_url = ws_url
        self.token = token
        self.on_data = on_data
        self.ws = None
        self.thread = None
        self._stop = False

    def _on_message(self, ws, message):
        # Upstox sends binary protobuf messages; websocket-client delivers bytes.
        try:
            # message may be bytes or text; handle both.
            if isinstance(message, bytes):
                parsed = parse_feed_message(message)
                if parsed:
                    # parsed is a dict; call callback
                    self.on_data(parsed)
                else:
                    # optional: try decoding as utf-8 JSON
                    try:
                        js = json.loads(message.decode('utf-8'))
                        self.on_data(js)
                    except Exception:
                        pass
            else:
                # text message (JSON) — sometimes control messages may be JSON
                try:
                    js = json.loads(message)
                    self.on_data(js)
                except Exception:
                    pass
        except Exception as e:
            print("WebSocket parse error:", e)

    def _on_open(self, ws):
        print("WebSocket opened to", self.ws_url)

    def _on_error(self, ws, err):
        print("WebSocket error:", err)

    def _on_close(self, ws, close_status_code, close_msg):
        print("WebSocket closed", close_status_code, close_msg)

    def start(self):
        headers = [f"Authorization: Bearer {self.token}"]
        self.ws = websocket.WebSocketApp(self.ws_url,
                                         header=headers,
                                         on_message=self._on_message,
                                         on_open=self._on_open,
                                         on_error=self._on_error,
                                         on_close=self._on_close)
        self.thread = threading.Thread(target=self.ws.run_forever, kwargs={"ping_interval": 30, "ping_timeout": 10}, daemon=True)
        self.thread.start()

    def stop(self):
        self._stop = True
        try:
            if self.ws:
                self.ws.close()
        except Exception:
            pass

    def send_subscribe(self, instrument_keys, mode="option_greeks"):
        """
        Subscribe to instrument keys. mode can be 'full' or 'option_greeks' depending on proto support.
        Must be called after connection is open. The feed expects a JSON subscribe message:
          { "guid": "<uuid>", "method":"sub", "data": { "instrumentKeys": ["NSE_FO|123"], "mode": "full" } }
        """
        if not self.ws:
            raise RuntimeError("WebSocket not connected")
        sub = {
            "guid": "sub_" + str(int(time.time())),
            "method": "sub",
            "data": {"instrumentKeys": instrument_keys, "mode": mode}
        }
        try:
            self.ws.send(json.dumps(sub))
        except Exception as e:
            print("Failed to send subscribe:", e)

# ====== Background manager thread ======
class BackgroundManager(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.ws_client = None
        self._stop = False

    def stop(self):
        self._stop = True
        if self.ws_client:
            self.ws_client.stop()

    def run(self):
        while not self._stop:
            token = DATA.get("token")
            if not token:
                time.sleep(1)
                continue

            now = ist_now()
            today = now.date().isoformat()
            # 1) If it's selection time and selection not yet done for today -> choose instruments
            if now.time() >= SELECT_TIME and DATA.get("selected_date") != today:
                try:
                    out = choose_today_instruments(token)
                    with DATA_LOCK:
                        DATA["selected_date"] = today
                        DATA["expiry"] = out["expiry"]
                        DATA["instrument_keys"] = out["selected_instruments"]
                        DATA["series"] = defaultdict(list)
                    print(f"[{today}] Selected {len(out['selected_instruments'])} instruments for expiry {out['expiry']} (ATM {out['underlying_price']})")
                except Exception as e:
                    print("Selection error:", e)

            # 2) Polling: ensure websocket connected & subscribed, then parse incoming feed
            if DATA.get("instrument_keys") and START_TIME <= now.time() <= STOP_TIME:
                try:
                    # ensure feeder URL exists
                    if not DATA.get("feeder_url"):
                        try:
                            url = get_marketdata_authorized_url(token)
                            DATA["feeder_url"] = url
                            print("Feeder URL:", url)
                        except Exception as e:
                            print("Failed to get feeder URL:", e)
                            time.sleep(2)
                            continue

                    if not self.ws_client:
                        # start ws
                        self.ws_client = UpstoxWS(DATA["feeder_url"], token, on_ws_data)
                        self.ws_client.start()
                        # give it a moment to connect
                        time.sleep(1)
                        # subscribe to our instruments in 'option_greeks' mode
                        self.ws_client.send_subscribe(DATA["instrument_keys"], mode="option_greeks")
                        print("Subscribed to instruments via ws")

                    # At this point incoming ws messages will call on_ws_data() which appends to DATA['series']
                    # We simply sleep here; data is event-driven
                    time.sleep(POLL_INTERVAL_SECONDS)
                except Exception as e:
                    print("Polling loop error:", e)
                    # drop ws client and retry
                    try:
                        if self.ws_client:
                            self.ws_client.stop()
                    except Exception:
                        pass
                    self.ws_client = None
                    time.sleep(2)
            else:
                # outside market window
                time.sleep(2)

# ====== Selection logic (choose 5 ITM & 5 OTM) ======
def choose_today_instruments(token):
    """
    1. Fetch option contracts for underlying (v2)
    2. Determine nearest expiry (or leave expiry selection dynamic)
    3. Choose ITM/OTM strikes around ATM using underlying LTP from LTP endpoint
    4. Return dict containing selected instrument_keys (CE + PE)
    """
    underlying_key = DATA.get("underlying_key", UNDERLYING_KEY_DEFAULT)
    # 1) get option contracts list
    contracts = fetch_option_contracts(token, underlying_key)
    if not contracts:
        raise RuntimeError("No option contracts returned from fetch_option_contracts()")
    df = pd.DataFrame(contracts)
    # normalise expiry to YYYY-MM-DD strings
    df["expiry_dt"] = pd.to_datetime(df["expiry"]).dt.date
    today = ist_now().date()
    future_expiries = sorted([d for d in df["expiry_dt"].unique() if d >= today])
    if not future_expiries:
        raise RuntimeError("No upcoming expiries found in contract list")
    expiry_choice = str(future_expiries[0])  # nearest expiry by default

    # 2) filter to chosen expiry
    df_exp = df[df["expiry_dt"] == pd.to_datetime(expiry_choice).date()]

    # 3) get underlying LTP
    # Upstox often exposes underlying as instrument_key like 'NSE_INDEX|Nifty 50' or 'NSE_INDEX|NIFTY 50'
    underlying_ltp_map = fetch_ltp_map(token, [underlying_key])
    underlying_price = underlying_ltp_map.get(underlying_key)
    if underlying_price is None:
        # try to guess a numeric spot by looking at index entries in contracts (rare fallback)
        underlying_price = float(df_exp['strike_price'].median())
    # 4) choose strikes
    strikes_ce = sorted(df_exp[df_exp["instrument_type"] == "CE"]["strike_price"].unique())
    strikes_pe = sorted(df_exp[df_exp["instrument_type"] == "PE"]["strike_price"].unique())
    # find ATM index for CE strikes
    def pick_side(strikes, is_ce=True):
        if not strikes:
            return []
        atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - underlying_price))
        low = strikes[max(0, atm_idx - ITM_COUNT): atm_idx]      # strikes below ATM (ITM for CE)
        high = strikes[atm_idx+1: atm_idx+1 + OTM_COUNT]         # strikes above ATM (OTM for CE)
        # for CE we want ITM=lower, OTM=higher; for PE the logic is symmetric but reversed roles
        return list(reversed(low)) + high

    selected_ce_strikes = pick_side(strikes_ce, is_ce=True)
    selected_pe_strikes = pick_side(strikes_pe, is_ce=False)

    # select instrument keys using strike & instrument_type filtering
    selected_instrument_keys = []
    # pick one instrument_key per strike per option type (some contract lists may have duplicates — pick first)
    for s in selected_ce_strikes:
        row = df_exp[(df_exp["strike_price"] == s) & (df_exp["instrument_type"] == "CE")]
        if not row.empty:
            selected_instrument_keys.append(row.iloc[0]["instrument_key"])
    for s in selected_pe_strikes:
        row = df_exp[(df_exp["strike_price"] == s) & (df_exp["instrument_type"] == "PE")]
        if not row.empty:
            selected_instrument_keys.append(row.iloc[0]["instrument_key"])

    return {
        "underlying_price": underlying_price,
        "selected_instruments": selected_instrument_keys,
        "selected_ce_strikes": selected_ce_strikes,
        "selected_pe_strikes": selected_pe_strikes,
        "expiry": expiry_choice
    }

# ====== Callback invoked when WS yields parsed dict ======
def on_ws_data(parsed_dict):
    """
    parsed_dict is result of MessageToDict(msg) or JSON decode — structure varies by proto version.
    We must inspect to find option_chain/option_greeks-like keys. Community examples indicate
    that decoded messages contain 'option_chain' or 'option_greeks' or similar blocks.
    We'll attempt to traverse and find 'instrumentKey' and greek fields.
    """
    try:
        # heuristics: if parsed_dict has 'feeds' or 'data' or 'payload' keys, drill in
        # Convert nested dict to simplified records where possible
        # Many MessageToDict outputs include nested keys; search for 'instrument_key' in dict
        def find_instruments(d):
            found = []
            if isinstance(d, dict):
                # direct mapping keys
                if 'instrument_key' in d or 'instrumentKey' in d:
                    found.append(d)
                else:
                    for v in d.values():
                        found.extend(find_instruments(v))
            elif isinstance(d, list):
                for item in d:
                    found.extend(find_instruments(item))
            return found

        matches = find_instruments(parsed_dict)
        ts = ist_now().strftime("%Y-%m-%d %H:%M:%S")
        with DATA_LOCK:
            for item in matches:
                ik = item.get("instrument_key") or item.get("instrumentKey")
                if not ik:
                    continue
                # read expected fields (field names may differ slightly)
                ltp = item.get("ltp") or item.get("ltpc") or item.get("ltp_value") or None
                iv = item.get("iv") or item.get("implied_volatility") or None
                delta = item.get("delta")
                gamma = item.get("gamma")
                vega = item.get("vega")
                theta = item.get("theta")
                rho = item.get("rho")
                # append a record
                DATA["series"][ik].append({
                    "ts": ts, "ltp": ltp, "iv": iv, "delta": delta,
                    "gamma": gamma, "vega": vega, "theta": theta, "rho": rho
                })
    except Exception as exc:
        print("on_ws_data error:", exc)

# ====== Streamlit UI ======
st.set_page_config(page_title="NIFTY Realtime Greeks (Upstox WebSocket V3)", layout="wide")
st.title("NIFTY Realtime Greeks (WebSocket V3 / protobuf)")

col1, col2 = st.columns([3,1])
with col1:
    token_input = st.text_input("Paste Upstox ACCESS TOKEN (daily)", type="password", value=DATA.get("token") or "")
with col2:
    if st.button("Save token"):
        if token_input and len(token_input) > 10:
            with DATA_LOCK:
                DATA["token"] = token_input.strip()
            st.success("Token saved to memory (not persisted). Background fetcher will use it.")
        else:
            st.error("Paste a valid token string.")

# Underlying and force selection controls
underlying = st.text_input("Underlying instrument key", value=DATA.get("underlying_key", UNDERLYING_KEY_DEFAULT))
if underlying != DATA.get("underlying_key"):
    with DATA_LOCK:
        DATA["underlying_key"] = underlying

if st.button("Force selection now (test)"):
    tok = DATA.get("token")
    if not tok:
        st.error("Save token first.")
    else:
        try:
            out = choose_today_instruments(tok)
            with DATA_LOCK:
                DATA["selected_date"] = ist_now().date().isoformat()
                DATA["expiry"] = out["expiry"]
                DATA["instrument_keys"] = out["selected_instruments"]
                DATA["series"] = defaultdict(list)
            st.success(f"Selected {len(out['selected_instruments'])} instruments for expiry {out['expiry']}")
        except Exception as e:
            st.error(f"Selection failed: {e}")

# Start background manager
if 'bg_mgr' not in st.session_state:
    st.session_state.bg_mgr = BackgroundManager()
    st.session_state.bg_mgr.start()
    time.sleep(0.2)

# Show summary metrics / table
with DATA_LOCK:
    selected_instruments = list(DATA.get("instrument_keys", []))
    expiry = DATA.get("expiry")
    selected_date = DATA.get("selected_date")

colA, colB, colC = st.columns(3)
colA.metric("Selected Date", selected_date or "-")
colB.metric("Expiry", expiry or "-")
colC.metric("Total instruments", len(selected_instruments))

st.subheader("Live snapshot (latest per instrument)")
rows = []
with DATA_LOCK:
    for ik in selected_instruments:
        hist = DATA["series"].get(ik, [])
        latest = hist[-1] if hist else {}
        rows.append({
            "instrument_key": ik,
            "ltp": latest.get("ltp"),
            "iv": latest.get("iv"),
            "delta": latest.get("delta"),
            "gamma": latest.get("gamma"),
            "vega": latest.get("vega"),
            "theta": latest.get("theta"),
            "rho": latest.get("rho"),
            "last_ts": latest.get("ts")
        })
df_live = pd.DataFrame(rows)
st.dataframe(df_live)

# Charts
st.subheader("Live Greeks charts")
tab1, tab2, tab3, tab4 = st.tabs(["Delta","Gamma","Vega","Theta(per day)"])

def plot_greek(greek_key, container):
    if not selected_instruments:
        container.info("No instruments selected yet.")
        return
    # build time-indexed dataframe
    recs = []
    with DATA_LOCK:
        for ik in selected_instruments:
            for r in DATA["series"].get(ik, []):
                recs.append({"ts": r["ts"], "instrument": ik, greek_key: (r.get(greek_key) or None)})
    if not recs:
        container.info("Waiting for feed data... (open market & correct token & subscribed instruments)")
        return
    df = pd.DataFrame(recs)
    df["ts_dt"] = pd.to_datetime(df["ts"])
    pivot = df.pivot_table(index="ts_dt", columns="instrument", values=greek_key)
    pivot = pivot.fillna(method="ffill").fillna(method="bfill")
    container.line_chart(pivot)

with tab1:
    plot_greek("delta", st)
with tab2:
    plot_greek("gamma", st)
with tab3:
    plot_greek("vega", st)
with tab4:
    plot_greek("theta", st)

st.caption("Notes: If protobuf parsing fails, confirm you generated feeds_proto/MarketDataFeedV3_pb2.py using protoc from Upstox's MarketDataFeedV3.proto. If messages are not arriving, ensure your token has 'marketdata.read' scope and that you retrieved an authorized ws URL via /v3/feed/market-data-feed/authorize. See logs in server for details.")
