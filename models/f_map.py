from torch import nn
hidden_units = 500

f_map = nn.Sequential(
        nn.Linear(2, hidden_units),
        nn.ReLU(True),
        nn.Linear(hidden_units, 2))
