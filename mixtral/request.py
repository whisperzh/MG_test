# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import sys
import json
import requests
import re
import os
from transformers import AutoTokenizer
 
if __name__ == "__main__":
    prompt =  "hi"
    b = "this is a cat"
    url = "localhost:5000"
    url = 'http://' + url + '/api'
    headers = {'Content-Type': 'application/json'}
    data = {"prompts":[prompt , b], "tokens_to_generate": 1}
    response = requests.put(url, data=json.dumps(data), headers=headers)

    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.json()['message']}")
    else:
        print("Megatron Response: ")
        print(response.json()['text'][0])
