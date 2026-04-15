import torch
import torch.optim as optim
import config
import random
from model import DQN


class DQNAgent:
    def __init__(self, state_size, action_size, replay_buffer):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = replay_buffer
        
        self.policy_net = DQN(state_size, action_size)

        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        #Disables Training Behaviors: It turns off specific |
        #layers used only during training (like Dropout or Batch Normalization) 
        # that would otherwise add randomness to the output.


        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        self.epsilon = config.EPSILON_START
        self.epsilon_end = config.EPSILON_END
        self.epsilon_decay = config.EPSILON_DECAY


    def action_selection(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size-1)
        
        state = torch.FloatTensor(state).unsqueeze(0)

        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state)
        self.policy_net.train()

        return q_values.argmax(1)



    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


    def learn(self):
        if self.memory.__len__() < config.BATCH_SIZE:
            return
        
        states, actions, rewards, next_states, reward = self.memory.sample(config.BATCH_SIZE)

        
        



