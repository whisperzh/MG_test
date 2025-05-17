
from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np

def format_prompt(sample):
    ctx = sample["ctx"].strip()
    endings = [e.strip() for e in sample["endings"]]
    prompt = f"Context: {ctx}\n"
    for i, ending in enumerate(endings):
        prompt += f"{chr(65+i)}. {ending}\n"
    prompt += "Answer:"
    return prompt

def analyze_hellaswag(split="validation", max_samples=1000):
    dataset = load_dataset("hellaswag", split=split, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("/home/ec2-user/CodeSpace/download/models/Mixtral-8x7B-v0.1/")
    if max_samples==None:
        max_samples = len(dataset)
    lengths = []
    for i in range(min(max_samples, len(dataset))):
        prompt = format_prompt(dataset[i])
        tokens = tokenizer(prompt, return_tensors="pt")
        lengths.append(tokens.input_ids.shape[1])

    lengths = np.array(lengths)
    print(f"📊 Stats on HellaSwag ({split} split, {len(lengths)} samples):")
    print(f"  Max length : {lengths.max()}")
    print(f"  Min length : {lengths.min()}")
    print(f"  Avg length : {lengths.mean():.2f}")
    print(f"  P90 length : {np.percentile(lengths, 90):.2f}")
    print(f"  P95 length : {np.percentile(lengths, 95):.2f}")

    # 可视化直方图
    plt.figure(figsize=(8, 5))
    plt.hist(lengths, bins=30, color='skyblue', edgecolor='black')
    plt.title(f"HellaSwag {split} Prompt Token Lengths")
    plt.xlabel("Token Count")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"hellaswag_token_length_{split}.png")


# 运行统计
analyze_hellaswag(split="validation", max_samples=None)
