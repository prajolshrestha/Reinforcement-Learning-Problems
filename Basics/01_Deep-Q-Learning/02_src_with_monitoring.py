import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

# Environment setup 
grid_size = 5
num_actions = 4
goal_state = (4,4)
obstacles = [(1,1), (2,2), (3,3)]
actions = {0: (-1, 0),
           1: (1, 0),
           2: (0, -1),
           3: (0, 1)}

# Hyperparameters
LR = 0.001
GAMMA = 0.9
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
EPISODES = 1000
TARGET_UPDATE = 10
MEMORY_SIZE = 2000
MAX_STEPS = grid_size * grid_size * 2

device = "cuda" if torch.cuda.is_available() else "cpu"

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, output_size)
        )
    def forward(self, x):
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def select_action(state, epsilon, policy_net):
    if np.random.rand() < epsilon:
        return np.random.choice(num_actions)
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            return q_values.argmax().item()
        
def move(state, action):
    new_state = (state[0] + actions[action][0],
                 state[1] + actions[action][1])

    if (0 <= new_state[0] < grid_size and
        0 <= new_state[1] < grid_size and
        new_state not in obstacles):
        return new_state
    return state     

def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return 0
    
    transitions = memory.sample(BATCH_SIZE)
    batch = list(zip(*transitions))
    
    state_batch = torch.FloatTensor(np.array(batch[0])).to(device)
    action_batch = torch.LongTensor(batch[1]).unsqueeze(1).to(device)
    reward_batch = torch.FloatTensor(batch[2]).to(device)
    next_state_batch = torch.FloatTensor(np.array(batch[3])).to(device)
    done_batch = torch.BoolTensor(batch[4]).to(device)

    current_q_values = policy_net(state_batch).gather(1, action_batch)
    next_q_values = target_net(next_state_batch).max(1)[0].detach()
    expected_q_values = reward_batch + (GAMMA * next_q_values * ~done_batch)

    loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def plot_training_results(losses, rewards, steps):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title('Training Loss Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(rewards)
    plt.title('Episode Rewards Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 3, 3)
    plt.plot(steps)
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.tight_layout()
    plt.show()

def train():
    policy_net = DQN(2, num_actions).to(device)
    target_net = DQN(2, num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)

    epsilon = EPSILON
    episode_rewards = []
    episode_lengths = []
    episode_losses = []
    
    for episode in tqdm(range(EPISODES), desc="Training Episodes"):
        state = (0,0)
        done = False
        total_reward = 0
        steps = 0
        episode_loss = 0
        
        while not done and steps < MAX_STEPS:
            state_np = np.array(state)
            action = select_action(state_np, epsilon, policy_net)
            new_state = move(state, action)
            steps += 1
            
            if steps >= MAX_STEPS:
                reward = -10
                done = True
            elif new_state == goal_state:
                reward = 10
                done = True
            else:
                reward = -1

            memory.push(state_np, action, reward, np.array(new_state), done)
            total_reward += reward
            
            loss = optimize_model(memory, policy_net, target_net, optimizer)
            if loss:
                episode_loss += loss
                
            state = new_state

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_losses.append(episode_loss/steps if steps > 0 else 0)
        
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        
    return policy_net, episode_losses, episode_rewards, episode_lengths

def test_agent(policy_net):
    state = (0,0)
    path = [state]
    steps = 0
    
    with torch.no_grad():
        while state != goal_state and steps < MAX_STEPS:
            state_np = np.array(state)
            state_tensor = torch.FloatTensor(state_np).unsqueeze(0).to(device)
            action = policy_net(state_tensor).argmax().item()
            state = move(state, action)
            path.append(state)
            steps += 1
            
    return path

if __name__ == "__main__":
    policy_net, losses, rewards, steps = train()
    plot_training_results(losses, rewards, steps)
    optimal_path = test_agent(policy_net)
    print("\nOptimal path found:", optimal_path)