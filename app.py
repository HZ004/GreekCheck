import os
import json
import time
import threading
import requests
import streamlit as st
from datetime import datetime
from upstox_api.api import Upstox
from feeds_proto import MarketDataFeedV3_pb2 as pb
from websocket import create_connection
import random

# ------------------------
# CONFIG
# ------------------------
UPSTOX_API_KEY = os.getenv("UPSTOX_API_KEY")
UPSTOX_API_SECRET = os.getenv("UPSTOX_API_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI", "https://your-redirect-url.com")
TOKEN_FILE = "access_token.json"
INSTRUMENTS_FILE = "instruments.json"
MOCK_MODE = True  # Toggle this to False for live market

# ------------------------
# TOKEN HANDLING
# ------------------------
def save_token(token_data):
    with open(TOKEN_FILE, "w") as f:
        json.dump(token_data, f)

def load_token():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r") as f:
            return json.load(f)
    return None

# ------------------------
# INSTRUMENT FETCH
# ------------------------
def fetch_instruments(up):
    """Fetch and save instrument master list."""
    resp = up.get_master_contract("NSE_FO")
    instruments = []
    for scrip, details in resp.items():
        instruments.append(details)
    with open(INSTRUMENTS_FILE, "w") as f:
        json.dump(instruments, f)
    return instruments

def load_instruments():
    if os.path.exists(INSTRUMENTS_FILE):
        with open(INSTRUMENTS_FILE, "r") as f:
            return json.load(f)
    return []

def get_today_contracts(instruments):
    """Filter contracts for today."""
    today = datetime.now().strftime("%Y-%m-%d")
    expiries = sorted({i['expiry'] for i in instruments if 'expiry' in i})
    contracts_today = [i for i in instruments if i.get('expiry') == expiries[0]]
    return today, expiries[0], contracts_today[:5]  # pick 5 contracts

# ------------------------
# MOCK DATA GENERATOR
# ------------------------
def generate_mock_data(contracts):
    """Simulate Greeks values for testing."""
    mock_data = {}
    for c in contracts:
        mock_data[c['token']] = {
            "ltp": round(random.uniform(100, 1000), 2),
            "delta": round(random.uniform(-1, 1), 3),
            "gamma": round(random.uniform(0, 0.1), 4),
            "vega": round(random.uniform(0, 1), 3),
            "theta": round(random.uniform(-1, 0), 3)
        }
    return mock_data

# ------------------------
# WEBSOCKET FETCHER
# ------------------------
class MarketDataFetcher:
    def __init__(self, contracts):
        self.contracts = contracts
        self.ws_url = "wss://api-v2.upstox.com/feed/market-data-feed/v3"
        self.ws = None
        self.data = {}

    def connect(self):
        try:
            self.ws = create_connection(self.ws_url)
            st.session_state["status"] = "WebSocket Connected"
            self.subscribe()
        except Exception as e:
            st.error(f"WebSocket connection failed: {e}")

    def subscribe(self):
        payload = {
            "guid": "some-guid",
            "method": "sub",
            "data": {
                "mode": "full",
                "instrumentKeys": [c['token'] for c in self.contracts]
            }
        }
        try:
            self.ws.send(json.dumps(payload))
        except Exception as e:
            st.error(f"Failed to subscribe: {e}")

    def listen(self):
        while True:
            try:
                msg = self.ws.recv()
                self.handle_message(msg)
            except Exception as e:
                print("Error in websocket loop:", e)
                break

    def handle_message(self, msg):
        try:
            pb_msg = pb.FeedResponse()
            pb_msg.ParseFromString(msg)
            for feed in pb_msg.feeds:
                self.data[feed.instrumentKey] = {
                    "ltp": feed.marketFF.marketPrice,
                    "delta": getattr(feed, "delta", None),
                    "gamma": getattr(feed, "gamma", None),
                    "vega": getattr(feed, "vega", None),
                    "theta": getattr(feed, "theta", None)
                }
        except Exception as e:
            print("Failed to parse protobuf:", e)

    def start(self):
        self.connect()
        threading.Thread(target=self.listen, daemon=True).start()

# ------------------------
# STREAMLIT UI
# ------------------------
st.set_page_config(page_title="Upstox Options Greeks", layout="wide")
st.title("📈 Real-Time Options Greeks Dashboard")

# Load token
token = load_token()
if not token:
    st.warning("⚠️ No access token found. Please generate and save token manually.")
else:
    try:
        up = Upstox(UPSTOX_API_KEY, token["access_token"])
        st.success("✅ Access token loaded.")
    except Exception as e:
        st.error(f"Failed to authenticate: {e}")
        st.stop()

    # Load or fetch instruments
    instruments = load_instruments()
    if not instruments:
        st.info("Fetching instruments...")
        instruments = fetch_instruments(up)
        st.success("Instrument data fetched.")

    # Get today's contracts
    today, expiry, selected_contracts = get_today_contracts(instruments)
    st.subheader("Today's Selection & Latest Values")
    st.write(f"**Date:** {today}")
    st.write(f"**Nearest Expiry:** {expiry}")
    st.table(selected_contracts)

    # Start data feed
    if "fetcher" not in st.session_state:
        fetcher = MarketDataFetcher(selected_contracts)
        if not MOCK_MODE:
            fetcher.start()
        st.session_state["fetcher"] = fetcher
        st.session_state["status"] = "Live Mode" if not MOCK_MODE else "Mock Mode"
    else:
        fetcher = st.session_state["fetcher"]

    # Display Greeks
    st.subheader("Live Greeks")
    if MOCK_MODE:
        data = generate_mock_data(selected_contracts)
    else:
        data = fetcher.data

    if data:
        rows = []
        for token, vals in data.items():
            rows.append({
                "Token": token,
                "LTP": vals.get("ltp"),
                "Delta": vals.get("delta"),
                "Gamma": vals.get("gamma"),
                "Vega": vals.get("vega"),
                "Theta": vals.get("theta")
            })
        st.dataframe(rows)
    else:
        st.info("Waiting for live data...")
