import requests
import time

url = "http://127.0.0.1:8001/chat"
payload = {"query": "Does weed help with hangovers?"}

print("Waiting for server to be ready...")
for i in range(10):
    try:
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            print("Server is ready!")
            print(f"Response: {response.json()}")
            break
    except Exception as e:
        print(f"Attempt {i+1}: Server not ready yet ({e})")
        time.sleep(2)
else:
    print("Server failed to respond in time.")
