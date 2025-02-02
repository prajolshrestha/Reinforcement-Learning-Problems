import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from tqdm import tqdm

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
LR = 0.001  # Learning rate
GAMMA = 0.9
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
EPISODES = 200
TARGET_UPDATE = 10 # Update target network every 10 episodes
MEMORY_SIZE = 2000

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# DQN: Deep Q-Network
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

# Experience replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen= capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Initialize network
input_size = 2  # (x,y) coordinates
policy_net = DQN(input_size, num_actions).to(device)
target_net = DQN(input_size, num_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayBuffer(MEMORY_SIZE)

# Epsilon-greedy action selection
def select_action(state, epsilon):
    if np.random.rand() > epsilon:
        return np.random.choice(num_actions) # Explore
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            return q_values.argmax().item() # Exploit
        
def move(state, action):
    new_state = (state[0] + actions[action][0],
                 state[1] + actions[action][1])

    # boundary check
    if (0 <= new_state[0] < grid_size and
        0 <= new_state[1] < grid_size and
        new_state not in obstacles):
        return new_state

    return state     

# Training function
def optimize_model():
    """
        # Step 1: Policy Net (Current State Evaluation)
        State (0,0) → Policy Net → Q-values for all actions at (0,0)
                                [UP, DOWN, LEFT, RIGHT]
                                [2.1, 3.2, 1.5, 4.7]
                                ↑
                                We took RIGHT action (4.7)

        # Step 2: Target Net (Future State Evaluation)
        State (0,1) → Target Net → Q-values for all actions at (0,1)
                    (next state)  [UP, DOWN, LEFT, RIGHT]
                                [5.1, 4.8, 4.2, 6.0]
                                ↑
                                Best future value = 6.0

        # Step 3: Bellman Equation
        Expected Q-value = reward + GAMMA * best_future_value
                        = -1 + 0.9 * 6.0
                        = 4.4

        # Step 4: Learning
        Policy Net learns: Q((0,0), RIGHT) should be 4.4
                        (not 4.7 as it currently predicts)



        # Training Process: #####################################

        1. Policy Net's Current Guess:
        policy_net: "I think action RIGHT from (0,0) is worth 4.7"

        2. Target Net's Guidance:
        target_net: "From the next state (0,1), best future value is 6.0"
        reward: -1
        target_value = -1 + 0.9 * 6.0 = 4.4

        3. Policy Net Learning:
        policy_net: "I was wrong (4.7), should be 4.4"
        # Updates its weights to move closer to 4.4

        4. Every 10 episodes:
        target_net: *gets updated with policy_net's learned weights*
        # Like a teacher learning from the student's discoveries
    """
    if len(memory) < BATCH_SIZE:
        return
    
    # Sample batch from replay buffer
    # 2. Random sampling for training
    transitions = memory.sample(BATCH_SIZE)
    # 3. Training with random batch
    batch = list(zip(*transitions))

    # Convert to tensors
    state_batch = torch.FloatTensor(np.array(batch[0])).to(device)
    action_batch = torch.FloatTensor(np.array(batch[1])).to(device)
    reward_batch = torch.FloatTensor(np.array(batch[2])).to(device)
    next_state_batch = torch.FloatTensor(np.array(batch[3])).to(device)
    done_batch = torch.FloatTensor(np.array(batch[4])).to(device)

    # compute Q-value using trained DQN
    # Example:  When we're in state (0,0) and take action RIGHT:
    # current_state = (0,0)
    # action = RIGHT
    # next_state = (0,1)
    # 1. Policy net estimates Q-value for the CURRENT state-action pair
    current_q_values = policy_net(state_batch).gather(1, action_batch)
    # This tells us: "How good is it to take RIGHT action from (0,0)?"

    # Compute V(s_{t+1}) using target network
    ## 2. Target net estimates FUTURE value from the NEXT state
    next_q_values = target_net(next_state_batch).max(1)[0].detach()
    # This tells us: "What's the best possible future value from (0,1)?"

    # 3. Bellman equation combines immediate reward with discounted future value
    expected_q_values = reward_batch + (GAMMA * next_q_values * ~done_batch)
    # Total value = immediate reward + (discount * best future value)

    # Compute loss
    loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(1))

    # Optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



# Training loop
epsilon = EPSILON
for episode in range(EPISODES):
    state = (0,0)
    done = False

    while not done:
        state_np = np.array(state) # we need to pass it to the network
        action = select_action(state_np, epsilon)
        new_state = move(state, action)

        if new_state == goal_state:
            reward = 10
            done = True
        else:
            reward = -1

        # Store transition in memory
        # 1. Storing experiences as we play
        memory.push(state, action, reward, np.array(new_state), done)

        # Move to next step
        state = new_state

        # optimize policy network
        optimize_model()

    # update target network
    if episode % TARGET_UPDATE == 0:
        print(episode)
        target_net.load_state_dict(policy_net.state_dict())
    
    # decay epsilon
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)


# Test the trained model
state = (0,0)
path = [state]
with torch.no_grad():
    while state != goal_state:
        state_np = np.array(state)
        state_tensor = torch.FloatTensor(state_np).unsqueeze(0).to(device)
        action = policy_net(state_tensor).argmax().item()
        state = move(state, action)
        path.append(state)

print("Optimal path:")
print(path)  

