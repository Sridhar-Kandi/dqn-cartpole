import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import config

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, config.HIDDEN_SIZE)
        self.fc2 = nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE)
        self.fc3 = nn.Linear(config.HIDDEN_SIZE, action_size)

    def forward(self, input):
        output = F.relu(self.fc1(input))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output
    

if __name__ == "__main__":
    dummy_state = torch.rand(1, 4)
    dqn = DQN(4, 2)
    output = dqn(dummy_state)
    print(output)
    print(output.shape)