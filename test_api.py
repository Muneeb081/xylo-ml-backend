import requests
import time
import json
import sys

# Wait for the server to be ready
url_health = "http://localhost:8000/health"
print("Waiting for API server to start...")
for i in range(15):
    try:
        res = requests.get(url_health, timeout=2)
        if res.status_code == 200:
            print("API is up and running!")
            break
    except requests.exceptions.RequestException:
        pass
    time.sleep(2)
else:
    print("API server failed to start in time.")
    sys.exit(1)

# Test the Firebase Stream endpoint
url_firebase = "http://localhost:8000/api/v1/stream/firebase"
payload = {
    "house_id": 16,
    "timestamp": "2024-05-01T14:30:00",
    "data": {
        "Usage_kW": 3.8,
        "kitchen": 1.5,
        "AC_DR_kW": 2.0,     # Should be mapped to 'ac'
        "unknown_fan": 0.2,
        "n_acs": 2,
        "n_people": 4
    }
}

print(f"\nSending payload to {url_firebase}:")
print(json.dumps(payload, indent=2))

try:
    response = requests.post(url_firebase, json=payload)
    print(f"\nResponse Code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error testing API: {e}")
