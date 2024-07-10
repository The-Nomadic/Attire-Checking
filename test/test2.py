import requests

# Send the request
with open('eight.png', 'rb') as f:
    resp = requests.post("https://get-prediction-ixifovjelq-an.a.run.app", files={'file': f})

# Print response status code
print("Status Code:", resp.status_code)

# Print response content
print("Response Content:", resp.text)

# Attempt to parse the response as JSON if the status code indicates success
if resp.status_code == 200:
    try:
        json_resp = resp.json()
        print("JSON Response:", json_resp)
    except requests.exceptions.JSONDecodeError:
        print("Error: Response is not in JSON format.")
else:
    print("Error: Received unexpected status code.")
