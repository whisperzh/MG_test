import torch
for i in range(0,4):
    a = torch.load(f"{i}.pt")
    b = torch.load(f"{i}_REPLICATE.pt")
    print(torch.equal(a, b))  # True if completely equal
