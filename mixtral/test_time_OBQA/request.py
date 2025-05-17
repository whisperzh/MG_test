# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import sys
import json
import requests
import pandas as pd
import re
import os
from transformers import AutoTokenizer
from datasets import load_dataset
if __name__ == "__main__":
    # control the input length
    url = "localhost:5000"
    url = 'http://' + url + '/api'
    headers = {'Content-Type': 'application/json'}
    dataset = load_dataset("openbookqa", "main")
    prompts = []
    sample_num = 1000
    sample_num = min(sample_num, len(dataset["train"]))
    print("Sample num",sample_num )
    for i in range(0,sample_num):
        sample = dataset["train"][i]
        question = sample["question_stem"]
        choices = sample["choices"]["text"]
        labels = sample["choices"]["label"]
        answer_key = sample["answerKey"]
        prompt = f"Question: {question}\n"
        for label, choice in zip(labels, choices):
            prompt += f"{label}. {choice}\n"
        prompt += "Answer:"
        prompts.append(prompt)
    for i in range(0,sample_num):
        print(f"=================={i}==================")
        data = {"prompts":[prompts[i] ], "tokens_to_generate": 1}
        response = requests.put(url, data=json.dumps(data), headers=headers)

        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.json()['message']}")
        else:
            print("Megatron Response: ")
            print(response.json()['text'][0])
