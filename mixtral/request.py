# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import sys
import json
import requests
import re
import os
from transformers import AutoTokenizer
 
if __name__ == "__main__":
    sys_prompt =  "How are you?"
    # user_prompt = """
    # Generate a detailed analysis of the impact of artificial intelligence on modern healthcare, covering at least five key areas such as diagnostics, treatment personalization,
    # robotic surgery, drug discovery, and patient monitoring. Include real-world examples where AI has significantly improved outcomes, potential risks or ethical concerns, and 
    # future trends. Additionally, compare the adoption rates of AI in healthcare between developed and developing nations, and suggest policy recommendations to bridge the gap. 
    # The response should be well-structured, data-driven, and cite reputable sources if possible.
    # """
    
    user_prompt="Hi"
    
    url = "localhost:5000"
    url = 'http://' + url + '/api'
    headers = {'Content-Type': 'application/json'}
    data = {"prompts":[sys_prompt , user_prompt,"See you next time."], "tokens_to_generate": 4}
    
    response = requests.put(url, data=json.dumps(data), headers=headers)

    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.json()['message']}")
    else:
        print("Megatron Response: ")
        print(response.json()['text'][0])
