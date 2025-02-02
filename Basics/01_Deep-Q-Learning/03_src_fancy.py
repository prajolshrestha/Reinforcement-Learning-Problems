import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, action_size)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        
        self.memory = deque(maxlen=2000)
        
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            act_values = self.model(state)
            return torch.argmax(act_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor([data[0] for data in minibatch]).to(self.device)
        actions = torch.LongTensor([data[1] for data in minibatch]).to(self.device)
        rewards = torch.FloatTensor([data[2] for data in minibatch]).to(self.device)
        next_states = torch.FloatTensor([data[3] for data in minibatch]).to(self.device)
        dones = torch.FloatTensor([data[4] for data in minibatch]).to(self.device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class GridWorldEnv:
    def __init__(self, grid_size=3):
        self.grid_size = grid_size
        self.state_size = 2
        self.action_size = 4
        self.obstacles = [(1,1)]
        self.goal_state = (2,2)
        self.actions = {
            0: (-1,0),
            1: (1,0),
            2: (0,-1),
            3: (0,1)
        }

    def reset(self):
        self.state = (0,0)
        return np.array(self.state)

    def step(self, action):
        new_state = (
            self.state[0] + self.actions[action][0],
            self.state[1] + self.actions[action][1]
        )

        if (0 <= new_state[0] < self.grid_size and 
            0 <= new_state[1] < self.grid_size and 
            new_state not in self.obstacles):
            self.state = new_state
        
        if self.state == self.goal_state:
            reward = 10
            done = True
        else:
            reward = -1
            done = False

        return np.array(self.state), reward, done

def train_dqn():
    env = GridWorldEnv()
    agent = DQNAgent(state_size=2, action_size=4)
    
    num_episodes = 100
    target_update_frequency = 10
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        if episode % target_update_frequency == 0:
            agent.update_target_model()
        
        if episode % 5 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    return agent

def test_agent(agent):
    env = GridWorldEnv()
    state = env.reset()
    path = [tuple(state)]
    
    while True:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            action = torch.argmax(agent.model(state_tensor)).item()
        
        state, reward, done = env.step(action)
        path.append(tuple(state))
        
        if done:
            break
    
    print("Optimal path found:", path)
    return path

if __name__ == "__main__":
    trained_agent = train_dqn()
    test_agent(trained_agent)