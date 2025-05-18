import requests

ride = {
    "PULocationID": 10,
    "DOLocationID": 50,
    "trip_distance": 40
}

response = requests.post('http://127.0.0.1:9696/predict', json=ride)
print("Status code:", response.status_code)
print("Response text:", response.text)
# Only try to parse JSON if the response is OK
if response.headers.get("Content-Type") == "application/json":
    print(response.json())
else:
    print("Response is not JSON.")