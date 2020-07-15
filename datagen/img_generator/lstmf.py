import torch
import torch.nn as nn
import numpy as np

class LstmNet(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(LstmNet, self).__init__()
        
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self. output_size = output_size

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
  
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
  
"""print(device)

input_size = 4
hidden_size = 7
num_layers = 16
output_size = 4 
batch_size = 16


model = LstmNet(input_size, output_size, hidden_size, num_layers)
hidden = model.init_hidden(batch_size)
model.to(device)

print(model)"""