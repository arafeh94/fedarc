import torch

device = torch.device("cuda:1000")
print(device.__repr__())
