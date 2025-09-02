import os
import requests
import streamlit as st
import pandas as pd
from streamlit_autorefresh import st_autorefresh

# 1. Setup
API_KEY = os.getenv("UPSTOX_API_KEY")
REDIRECT_URI = os.getenv("UPSTOX_REDIRECT_URI")
AUTH_URL = f"https://api.upstox.com/v2/login/authorization/dialog?client_id={API_KEY}&redirect_uri={REDIRECT_URI}&response_type=code"

st.set_page_config(page_title="Options Dashboard", layout="wide")
st_autorefresh(interval=30_000, key="refresh")
st.sidebar.info("🔄 Auto-refresh every 30 seconds")

# 2. Authenticate
if "access_token" not in st.session_state:
    st.session_state.access_token = None

st.title("🔹 Options Dashboard")
if not st.session_state.access_token:
    st.markdown(f"[Login to Upstox]({AUTH_URL})")
    code = st.text_input("Enter the `code` from the redirect URL")
    if st.button("Get Access Token") and code:
        resp = requests.post(
            "https://api.upstox.com/v2/login/authorization/token",
            data={
                "code": code,
                "client_id": API_KEY,
                "client_secret": os.getenv("UPSTOX_API_SECRET"),
                "redirect_uri": REDIRECT_URI,
                "grant_type": "authorization_code"
            },
        )
        data = resp.json()
        st.session_state.access_token = data.get("access_token")
        if st.session_state.access_token:
            st.success("Access token saved!")

# 3. Helper Functions
def get_headers():
    return {"Authorization": f"Bearer {st.session_state.access_token}", "Accept": "application/json"}

@st.cache_data(ttl=30)
def get_spot():
    r = requests.get("https://api.upstox.com/v2/market/quote", headers=get_headers(), params={"symbol":"NSE_INDEX|Nifty 50"})
    return float(r.json()["data"]["NSE_INDEX|Nifty 50"]["last_price"])

@st.cache_data(ttl=300)
def get_expiries():
    r = requests.get("https://api.upstox.com/v2/option/contract", headers=get_headers(), params={"symbol":"NIFTY50"})
    exps = sorted({c["expiryDate"] for c in r.json()["data"]})
    return exps[:4]

def get_contracts(expiry):
    r = requests.get("https://api.upstox.com/v2/option/contract", headers=get_headers(), params={"symbol":"NIFTY50","expiryDate":expiry})
    data = r.json()["data"]
    spot = get_spot()
    strikes = sorted({float(c["strikePrice"]) for c in data})
    atm = min(strikes, key=lambda x: abs(x - spot))
    idx = strikes.index(atm)
    sel = strikes[max(0,idx-5):idx+6]
    return [c for c in data if float(c["strikePrice"]) in sel], spot, atm

def fetch_greeks(keys):
    iks = ",".join(keys)
    r = requests.get(f"https://api.upstox.com/v3/market-quote/option-greek?instrument_key={iks}", headers=get_headers())
    return r.json().get("data", {})

# 4. Main Logic
if st.session_state.access_token:
    st.subheader("Today's Options Greek Snapshot")
    exps = get_expiries()
    expiry = st.selectbox("Select Expiry", exps)

    if expiry:
        contracts, spot, atm = get_contracts(expiry)
        st.write(f" Spot: {spot} | ATM Strike: {atm}")

        if contracts:
            symbols = [c["instrument_key"] for c in contracts]
            greeks_data = fetch_greeks(symbols)
            rows = []
            for c in contracts:
                key = c["instrument_key"]
                g = greeks_data.get(key, {})
                rows.append({
                    "Symbol": key,
                    "Strike": float(c["strikePrice"]),
                    "Type": c["instrument_type"],
                    "LTP": g.get("last_price"),
                    "IV": g.get("iv"),
                    "Delta": g.get("delta"),
                    "Gamma": g.get("gamma"),
                    "Theta": g.get("theta"),
                    "Vega": g.get("vega"),
                    "OI": g.get("oi"),
                })
            df = pd.DataFrame(rows).sort_values("Strike").reset_index(drop=True)

            def highlight_atm(r):
                return ["background-color: yellow"]*len(r) if abs(r["Strike"]-atm)<0.5 else [""]*len(r)

            calls = df[df["Type"]=="CE"]
            puts = df[df["Type"]=="PE"]

            st.markdown("###  Calls")
            st.dataframe(calls.style.apply(highlight_atm,axis=1), use_container_width=True)

            st.markdown("###  Puts")
            st.dataframe(puts.style.apply(highlight_atm,axis=1), use_container_width=True)
        else:
            st.warning("No contracts found for this expiry.")
