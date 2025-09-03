import time
import datetime
import requests

# Configuration
SERVICE_URL = "https://greekcheck-1.onrender.com/"  # Replace with your actual Render service URL

# IST timezone
IST_OFFSET = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

def keep_service_alive():
    print("Starting keep-alive pings...")
    while True:
        now = datetime.datetime.now(IST_OFFSET)
        if now.hour < 9 or (now.hour == 9 and now.minute < 10):
            print(f"Before active ping hours: {now.isoformat()}")
            break
        if now.hour > 15 or (now.hour == 15 and now.minute > 40):
            print(f"After active ping hours: {now.isoformat()}")
            break

        try:
            print(f"Pinging {SERVICE_URL} at {now.isoformat()}")
            response = requests.get(SERVICE_URL)
            print(f"Response status: {response.status_code}")
        except Exception as e:
            print(f"Exception during ping: {e}")

        time.sleep(300)  # Wait 15 minutes before next ping

if __name__ == "__main__":
    keep_service_alive()
