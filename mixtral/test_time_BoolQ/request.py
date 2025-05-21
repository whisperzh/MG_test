# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import sys
import json
import requests
import pandas as pd
import re
import random
import os
from transformers import AutoTokenizer
from datasets import load_dataset
if __name__ == "__main__":
    # control the input length
    sample_num =  1000
    split="validation"
    dataset =  load_dataset("boolq", split=split, trust_remote_code=True)
    sample_num = min(sample_num, len(dataset))
    print("Sample num",sample_num )
    prompts = []
    random.seed(42)
    indices = random.sample(range(len(dataset)), sample_num)
    print(indices[:10])
    for idx in indices:
        item = dataset[idx]
        question = item["question"].strip()
        passage = item["passage"].strip()
        answer = "yes" if item["answer"] else "no"

        prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer:"
        prompts.append(prompt)
     
    url = "localhost:5000"
    url = 'http://' + url + '/api'
    headers = {'Content-Type': 'application/json'}
    for i in range(0,sample_num) :
        print(f"=================={i}==================")
        data = {"prompts":[prompts[i] ], "tokens_to_generate": 1}
        response = requests.put(url, data=json.dumps(data), headers=headers)

        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.json()['message']}")
        else:
            print("Megatron Response: ")
            print(response.json()['text'][0])
