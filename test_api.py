import base64
import requests
import json

# Replace with your actual Invoke URL (Ensure it ends with /predict)
API_URL = "https://jff4nvcjo4z45q7sx4umdxcmia0rwofi.lambda-url.eu-north-1.on.aws/"
IMAGE_PATH = "test_image.png"

# Read and encode the image
with open(IMAGE_PATH, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

# Package the payload exactly as your Lambda function expects it
payload = {"image": encoded_string}
headers = {"Content-Type": "application/json"}

print("Sending image to AWS Cloud...")
response = requests.post(API_URL, json=payload, headers=headers)

# Print the final result
print("Status Code:", response.status_code)
print("Prediction Result:", json.dumps(response.json(), indent=2))