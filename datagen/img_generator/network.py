import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.drop_layer = nn.Dropout(p=0.0)
        self.fc1 = nn.Linear(17, 128)
        self.fc2 = nn.Linear(128, 96)
        self.fc3 = nn.Linear(96, 64)
        self.fc4 = nn.Linear(64, 48)
        self.fc5 = nn.Linear(48, 36)
        self.fc6 = nn.Linear(36, 24)
        self.fc7 = nn.Linear(24, 18)
        self.fc8 = nn.Linear(18, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc2(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc3(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc4(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc5(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc6(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc7(x))
        x = self.drop_layer(x)
        x = self.fc8(x)
        return x


class Net_Regressor(nn.Module):
    def __init__(self):
        super(Net_Regressor, self).__init__()
        self.drop_layer = nn.Dropout(p=0.15)
        self.fc1 = nn.Linear(11, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc2(x))
        x = self.drop_layer(x)
        x = self.fc3(x)
        return x