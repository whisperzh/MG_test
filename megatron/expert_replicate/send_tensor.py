import torch
import requests
import json
import time

# The URL you want to send the POST request to

baseurl="http://172.31.46.92:5000"
urlpost = baseurl + '/forward'
urlget = baseurl + '/ping'

# The data you want to send (can be a dictionary, list, etc.)
cuda_device="cuda:0"
tokens_per_expert = torch.tensor([32]*3, dtype=torch.int32).to(cuda_device)
dispatched_input = torch.randn(1, 32*3, 4096,
                                   dtype=torch.bfloat16).to(cuda_device)
data = {
    'dispatched_input': dispatched_input.tolist(),
    'tokens_per_expert': tokens_per_expert.tolist(),
    'layer':'0'
}


def post():
    # Send the POST request
    start=time.time()
    response = requests.post(
        urlpost,
        json=data,  # Convert data to JSON string

    )

    # Check if the request was successful
    if response.status_code == 200:
        end=time.time()
        print(f"{(end-start)*1000:.2f} ms used to process the data")
        
        print("Request successful!")
        print("Response:", response.json()["latency_ms"])  # Assuming response is JSON
    else:
        print(f"Request failed with status code: {response.status_code}")
        print("Response:", response.text)
def ping():
    response = requests.get(
        urlget,  # Convert data to JSON string

    )
    print("Response:", response.json())  # Assuming response is JSON

    

try:
    post()
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")