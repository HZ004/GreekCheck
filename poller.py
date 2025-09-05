import os
from datetime import datetime, time
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
import requests
import gspread
from google.oauth2.service_account import Credentials
from scipy.stats import norm
import time as pytime

# Google Sheets setup
SECRET_FILE_PATH = "/var/opt/render/secrets/service_account.json"
if os.path.exists(SECRET_FILE_PATH):
    SERVICE_ACCOUNT_FILE = SECRET_FILE_PATH
else:
    service_account_json = os.getenv("SERVICE_ACCOUNT_JSON")
    if not service_account_json:
        raise ValueError("Service account credentials not found. Set SERVICE_ACCOUNT_JSON env var or upload secret file.")
    tmp_path = "/tmp/service_account.json"
    with open(tmp_path, "w") as f:
        f.write(service_account_json)
    SERVICE_ACCOUNT_FILE = tmp_path

SPREADSHEET_NAME = "Upstox-Greeks"
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']

creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
gc = gspread.authorize(creds)
sheet = gc.open(SPREADSHEET_NAME).sheet1

def append_greeks_to_sheets(df: pd.DataFrame):
    existing = sheet.get_all_values()
    if not existing:
        sheet.append_row(list(df.columns))
    for _, row in df.iterrows():
        sheet.append_row(row.astype(str).tolist())

# Upstox API config and helpers
EXCHANGE = "NSE_INDEX"
SYMBOL = "Nifty 50"
STRIKES_TO_PICK = 5
UPSTOX_TOKEN = os.getenv("UPSTOX_TOKEN")
if not UPSTOX_TOKEN:
    raise Exception("UPSTOX_TOKEN environment variable not set")

HEADERS = {"Authorization": f"Bearer {UPSTOX_TOKEN}", "Accept": "application/json"}
API_OP_CONTRACTS = "https://api.upstox.com/v2/option/contract"
API_GREEKS = "https://api.upstox.com/v3/market-quote/option-greek"
API_LTP = "https://api.upstox.com/v3/market-quote/ltp"
IST = ZoneInfo("Asia/Kolkata")

def fetch_option_contracts():
    params = {"instrument_key": f"{EXCHANGE}|{SYMBOL}"}
    r = requests.get(API_OP_CONTRACTS, headers=HEADERS, params=params)
    response = r.json()
    if "data" not in response:
        raise RuntimeError(f"Failed fetching contracts: {response}")
    df = pd.DataFrame(response["data"])
    df["strike_price"] = df["strike_price"].astype(float)
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    return df

def fetch_spot_price(spot_instrument_key):
    r = requests.get(API_LTP, headers=HEADERS, params={"instrument_key": spot_instrument_key})
    response = r.json()
    key_in_data = spot_instrument_key.replace("|", ":")
    if "data" in response and key_in_data in response["data"]:
        return float(response["data"][key_in_data].get("last_price", 0))
    raise RuntimeError(f"Failed fetching spot price: {response}")

def get_nearest_expiry(contract_df):
    today = datetime.now(IST).date()
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
    ltp_map = {}
    for k, v in ltp_resp.items():
        token = v.get("instrument_token")
        if token:
            ltp_map[token] = v.get("last_price", np.nan)
    greeks_map = {}
    for k, v in greeks_resp.items():
        token = v.get("instrument_token")
        if token:
            greeks_map[token] = v
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
    now = datetime.now(IST)
    expiry = datetime.combine(expiry_date, time(15, 30))
    return max(1e-6, (expiry - now).total_seconds() / (365.0 * 86400))

def fallback_compute(contract, spot, ltp):
    K = contract["strike_price"]
    T = get_years_to_expiry(contract["expiry"])
    instrument_type = contract["instrument_type"]
    sigma = 0.2
    return black_scholes_greeks(spot, K, T, sigma, instrument_type)

def main_loop():
    contract_df = fetch_option_contracts()
    spot_price = fetch_spot_price(f"{EXCHANGE}|{SYMBOL}")

    start_poll = time(9, 20)
    end_poll = time(15, 20)

    while True:
        now = datetime.now(IST)
        now_time = now.time()
        if start_poll <= now_time <= end_poll:
            try:
                expiry = get_nearest_expiry(contract_df)
                sel_df = select_option_strikes(contract_df, spot_price, expiry, STRIKES_TO_PICK)
                filtered_df = sel_df[sel_df["strike_price"] % 100 == 0].sort_values(by=["instrument_type", "strike_price"])
                keys_monitored = list(filtered_df["instrument_key"])
                greek_data = poll_greeks_ltp(keys_monitored)
                timestamp = now.isoformat()
                records = []
                for _, contract in filtered_df.iterrows():
                    ikey = contract["instrument_key"]
                    gd = greek_data.get(ikey, {})
                    ltp = gd.get("ltp", float("nan"))
                    delta_val = gd.get("delta", None)
                    if delta_val in [None, ""]:
                        delta_val = fallback_compute(contract, spot_price, ltp).get("delta", float("nan"))
                    if contract["instrument_type"] == "PE" and isinstance(delta_val, (int, float)) and not pd.isna(delta_val):
                        delta_val = abs(delta_val)
                    gamma_val = gd.get("gamma", None)
                    if gamma_val in [None, ""]:
                        gamma_val = fallback_compute(contract, spot_price, ltp).get("gamma", float("nan"))
                    theta_val = gd.get("theta", None)
                    if theta_val in [None, ""]:
                        theta_val = fallback_compute(contract, spot_price, ltp).get("theta", float("nan"))
                    record = {
                        "timestamp": timestamp,
                        "instrument_key": ikey,
                        "ltp": ltp,
                        "delta": delta_val,
                        "gamma": gamma_val,
                        "theta": theta_val,
                        "strike_price": contract["strike_price"],
                        "instrument_type": contract["instrument_type"],
                        "expiry": str(contract["expiry"])
                    }
                    records.append(record)
                new_data_df = pd.DataFrame(records)
                append_greeks_to_sheets(new_data_df)
                print(f"Appended new data at {timestamp}")
            except Exception as e:
                print(f"Error during polling: {e}")
        else:
            print(f"Outside polling hours at {now.isoformat()}")
        pytime.sleep(60)  # Poll every 60 seconds

if __name__ == "__main__":
    main_loop()
