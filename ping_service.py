import os
import datetime
import requests
import sys

def main():
    SERVICE_URL = os.getenv('SERVICE_URL')
    if not SERVICE_URL:
        print('Error: SERVICE_URL environment variable not set.')
        sys.exit(1)

    IST_OFFSET = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

    now = datetime.datetime.now(IST_OFFSET)
    hour, minute = now.hour, now.minute
    print(f'Script run at {now.isoformat()}')
    try:
        if (hour == 9 and minute >= 10) or (10 <= hour < 15) or (hour == 15 and minute <= 40):
            print(f'Pinging service at IST time {now.isoformat()}')
            response = requests.get(SERVICE_URL, timeout=10)
            print(f'Ping status code: {response.status_code}')
        else:
            print(f'Outside active ping hours at {now.isoformat()}')
    except Exception as e:
        print(f'Ping failed: {e}')

if __name__ == '__main__':
    main()
