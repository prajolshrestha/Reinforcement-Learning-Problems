import numpy as np

# Define environment parameters
grid_size = 3
num_actions = 4
# Dictionary mapping action indices to direction vectors (row, col)
actions = {0: (-1,0),  # up    - decrease row
          1: (1,0),    # down  - increase row
          2: (0,-1),   # left  - decrease column
          3: (0,1)}    # right - increase column

# Define environment obstacles and goal
obstacles = [(1,1)]    # Center cell is blocked
goal_state = (2,2)     # Bottom-right corner is the goal

# obstacles = [(1,1), (2,2), (3,3)]
# goal_state = (4,4)

# Initialize Q-table with zeros: shape = (rows, cols, possible_actions)
Q = np.zeros((grid_size, grid_size, num_actions))

def choose_action(state):
    """
    Epsilon-greedy action selection:
    - With probability epsilon: choose random action (exploration)
    - With probability 1-epsilon: choose best action (exploitation)
    """
    if np.random.rand() < epsilon:
        return np.random.choice(num_actions)  # Exploration
    else:
        return np.argmax(Q[state[0], state[1], :])  # Exploitation - choose action with highest Q-value

def move(state, action):
    """
    Execute action from current state and return new state.
    Handles boundary checks and obstacle collisions.
    """
    new_state = (state[0] + actions[action][0], state[1] + actions[action][1])

    # Check if new state is valid (within grid and not an obstacle)
    if (0 <= new_state[0] < grid_size and 
        0 <= new_state[1] < grid_size and 
        new_state not in obstacles):
        return new_state
    
    return state  # If invalid move, stay in current state

# Hyperparameters
alpha = 0.1      # Learning rate: how much to update Q-values (0-1)
gamma = 0.9      # Discount factor: importance of future rewards (0-1)
epsilon = 1.0    # Initial exploration rate
epsilon_decay = 0.995  # Rate at which exploration decreases
epsilon_min = 0.01     # Minimum exploration rate
num_episodes = 300    # Number of training episodes

# Training loop
for episode in range(num_episodes):
    state = (0,0)  # Start state (top-left corner)
    done = False

    while not done:
        # 1. Choose and execute action
        action = choose_action(state)
        new_state = move(state, action)

        # 2. Get reward
        if state != goal_state:
            reward = -1  # Small penalty for each step
        else:
            reward = 10  # Large reward for reaching goal
            done = True

        # 3. Q-learning update formula
        # Q(s,a) = Q(s,a) + α[R + γ*max(Q(s',a')) - Q(s,a)]
        # where: s=state, a=action, R=reward, s'=new_state, α=learning rate, γ=discount factor
        
        # Q[state[0], state[1], action] += alpha * (
        #     reward + 
        #     gamma * np.max(Q[new_state[0], new_state[1], :]) - 
        #     Q[state[0], state[1], action]
        # )

        ###################################################
        # Before update
        old_q = Q[state[0], state[1], action] # What you previously thought this state-action was worth?
        future_q = np.max(Q[new_state[0], new_state[1], :]) # future reward: What you expect to get in the future?
        
        # Calculate TD error
        #
        # reward: What you get right now
        # γ max future Q-value: What you expect to get in the future
        # current Q-value: What you previously thought this state-action was worth
        td_error = reward + gamma * future_q - old_q # 

        # new Q
        new_q = old_q + alpha * td_error
        
        # Update Q-value
        Q[state[0], state[1], action] = new_q
        ####################################################
        state = new_state
    
    # Decay exploration rate
    # Training Flow:
    #Start → Mostly Exploration → Balanced → Mostly Exploitation → Converged Policy
    #        (high epsilon)     (learning)    (low epsilon)     (optimal actions)
    #
    # #Early Episodes (High ε ≈ 1.0): 100% chance of random action | Pure exploration phase | Agent tries many different paths 
    #Middle Episodes (Medium ε ≈ 0.5): 50% chance of random action | Balance between exploration and exploitation | Agent starts using learned knowledge while still exploring
    #Late Episodes (Low ε ≈ 0.01): 1% chance of random action | Mostly exploitation | Agent uses best known actions
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

print("Learned Q-table:")
print(Q)

# Test the learned policy
state = (0,0)
path = [state]
while state != goal_state:
    # Always choose the best action (pure exploitation)
    action = np.argmax(Q[state[0], state[1], :])
    state = move(state, action)
    path.append(state)

print("Optimal path found:", path)




