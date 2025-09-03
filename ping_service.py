import os
import datetime
import requests

SERVICE_URL = os.getenv("SERVICE_URL")  # Your Render Streamlit app URL

IST_OFFSET = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

def keep_alive_ping():
    now = datetime.datetime.now(IST_OFFSET)
    print(f"Pinging service at IST time {now.isoformat()}")
    try:
        response = requests.get(SERVICE_URL)
        print(f"Ping status code: {response.status_code}")
    except Exception as e:
        print(f"Ping failed: {e}")

def main():
    now = datetime.datetime.now(IST_OFFSET)
    hour, minute = now.hour, now.minute
    # Ping only between 09:10 and 15:40 IST
    if (hour == 9 and minute >= 10) or (10 <= hour < 15) or (hour == 15 and minute <= 40):
        keep_alive_ping()
    else:
        print(f"Outside active ping hours at {now.isoformat()}")

if __name__ == "__main__":
    main()
