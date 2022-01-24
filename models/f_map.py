from torch import nn
import torch

hidden_units = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

f_map = nn.Sequential(
        nn.Linear(2, hidden_units),
        nn.ReLU(True),
        nn.Linear(hidden_units, 2))

f_map = f_map.to(device)
