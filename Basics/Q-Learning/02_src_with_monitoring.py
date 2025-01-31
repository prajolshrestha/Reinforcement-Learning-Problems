import numpy as np
import matplotlib.pyplot as plt

# Define environment parameters
grid_size = 5
num_actions = 4
# Dictionary mapping action indices to direction vectors (row, col)
actions = {0: (-1,0),  # up    - decrease row
          1: (1,0),    # down  - increase row
          2: (0,-1),   # left  - decrease column
          3: (0,1)}    # right - increase column

# Define environment obstacles and goal
obstacles = [(1,1), (2,2), (3,3)]    # Center cell is blocked
goal_state = (4,4)     # Bottom-right corner is the goal

# Initialize Q-table with zeros: shape = (rows, cols, possible_actions)
Q = np.zeros((grid_size, grid_size, num_actions))

def choose_action(state, epsilon):
    """
    Epsilon-greedy action selection:
    - With probability epsilon: choose random action (exploration)
    - With probability 1-epsilon: choose best action (exploitation)
    """
    if np.random.rand() < epsilon:
        return np.random.choice(num_actions)  # Exploration
    else:
        return np.argmax(Q[state[0], state[1], :])  # Exploitation

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

def train_agent():
    # Hyperparameters
    alpha = 0.1      # Learning rate: how much to update Q-values (0-1)
    gamma = 0.9      # Discount factor: importance of future rewards (0-1)
    epsilon = 1.0    # Initial exploration rate
    epsilon_decay = 0.995  # Rate at which exploration decreases
    epsilon_min = 0.01     # Minimum exploration rate
    num_episodes = 1000    # Number of training episodes
    convergence_threshold = 0.00001  # Threshold for early stopping

    # Tracking variables
    q_changes = []    # Store the max change in Q-values for each episode
    avg_rewards = []  # Store average reward per episode
    episode_lengths = []  # Store number of steps per episode

    # Training loop
    for episode in range(num_episodes):
        state = (0,0)  # Start state
        done = False
        max_change = 0  # Track maximum Q-value change in this episode
        episode_reward = 0  # Track total reward for this episode
        steps = 0  # Track number of steps in this episode

        while not done:
            # 1. Choose and execute action
            action = choose_action(state, epsilon)
            new_state = move(state, action)

            # 2. Get reward
            if new_state == goal_state:
                reward = 10  # Large reward for reaching goal
                done = True
            else:
                reward = -1  # Small penalty for each step

            # Store old Q-value for change tracking
            old_q = Q[state[0], state[1], action]

            # 3. Q-learning update
            Q[state[0], state[1], action] += alpha * (
                reward + 
                gamma * np.max(Q[new_state[0], new_state[1], :]) - 
                Q[state[0], state[1], action]
            )

            # Track maximum Q-value change
            change = abs(Q[state[0], state[1], action] - old_q)
            max_change = max(max_change, change)

            # Update tracking variables
            episode_reward += reward
            steps += 1
            state = new_state

            # Optional: Add step limit to prevent infinite loops during training
            if steps > 100:  # Max steps per episode
                done = True

        # Store episode statistics
        q_changes.append(max_change)
        avg_rewards.append(episode_reward / steps)
        episode_lengths.append(steps)
        
        # Decay exploration rate
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}")
            print(f"Max Q-value change: {max_change:.6f}")
            print(f"Average reward: {avg_rewards[-1]:.2f}")
            print(f"Episode length: {steps}")
            print(f"Current epsilon: {epsilon:.3f}\n")

        # Early stopping check
        if max_change < convergence_threshold:
            print(f"Converged after {episode + 1} episodes!")
            break

    return q_changes, avg_rewards, episode_lengths

# Add smoothing to see trends more clearly
def plot_with_smoothing(data, window=10):
    smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
    plt.plot(smoothed, label='Smoothed')
    plt.plot(data, alpha=0.3, label='Raw')
    plt.legend()

def plot_training_results(q_changes, avg_rewards, episode_lengths):
    plt.figure(figsize=(15, 5))

    # Plot Q-value changes
    plt.subplot(1, 3, 1)
    plot_with_smoothing(q_changes)
    plt.title('Q-value Changes Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Max Q-value Change')

    # Plot average rewards
    plt.subplot(1, 3, 2)
    plt.plot(avg_rewards)
    plt.title('Average Reward Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')

    # Plot episode lengths
    plt.subplot(1, 3, 3)
    plt.plot(episode_lengths)
    plt.title('Episode Length Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Number of Steps')

    plt.tight_layout()
    plt.show()

def test_agent():
    """Test the learned policy and return the optimal path"""
    state = (0,0)
    path = [state]
    
    while state != goal_state:
        action = np.argmax(Q[state[0], state[1], :])  # Choose best action
        state = move(state, action)
        path.append(state)
        
        # Prevent infinite loops
        if len(path) > 100:
            print("Warning: Could not reach goal state")
            break
    
    return path

def visualize_q_learning_step():
    # Example state transition
    current_state = (0, 0)
    action = 3  # right
    new_state = (0, 1)
    reward = -1
    
    gamma = 0.9
    alpha = 0.1
    # Before update
    old_q = Q[current_state[0], current_state[1], action]
    future_q = np.max(Q[new_state[0], new_state[1], :])
    
    # Calculate TD error
    td_error = reward + gamma * future_q - old_q
    
    # Update Q-value
    Q[current_state[0], current_state[1], action] += alpha * td_error
    
    print(f"Step Details:")
    print(f"Current Q-value: {old_q:.2f}")
    print(f"Reward: {reward}")
    print(f"Discounted future value: {gamma * future_q:.2f}")
    print(f"TD Error: {td_error:.2f}")
    print(f"Q-value update: {alpha * td_error:.2f}")
    print(f"New Q-value: {Q[current_state[0], current_state[1], action]:.2f}")

def demonstrate_value_gradient():
    """
    Demonstrates how discount factor creates a value gradient from goal to start
    """
    # Simple 1D path from start to goal
    path_length = 5
    goal_reward = 10
    step_reward = -1
    
    # Calculate values with different discount factors
    gammas = [0.5, 0.9, 1.0]
    
    print("Value gradient from goal to start:")
    print("Position | γ=0.5  | γ=0.9  | γ=1.0")
    print("-" * 35)
    
    for pos in range(path_length-1, -1, -1):  # Start from goal, move backwards
        values = []
        for gamma in gammas:
            # Steps remaining to goal
            steps_to_goal = path_length - 1 - pos
            
            # Calculate discounted future value
            value = goal_reward * (gamma ** steps_to_goal)
            # Add intermediate step penalties
            for step in range(steps_to_goal):
                value += step_reward * (gamma ** step)
            values.append(value)
        
        position_label = "Goal" if pos == path_length-1 else f"Step {pos}"
        print(f"{position_label:8} | {values[0]:6.2f} | {values[1]:6.2f} | {values[2]:6.2f}")

def visualize_q_values():
    """
    Visualize Q-values for each state and action after learning
    """
    # Assuming Q-table is already trained
    print("\nFinal Q-values for each state:")
    print("Format: [UP, DOWN, LEFT, RIGHT]")
    print("-" * 50)
    
    for i in range(grid_size):
        for j in range(grid_size):
            if (i,j) == goal_state:
                print(f"State ({i},{j}) [GOAL]:")
            elif (i,j) in obstacles:
                print(f"State ({i},{j}) [OBSTACLE]:")
            else:
                print(f"State ({i},{j}):")
            
            q_values = Q[i,j]
            actions = ["UP   ", "DOWN ", "LEFT ", "RIGHT"]
            
            # Print each action's Q-value
            for action, q_value in zip(actions, q_values):
                print(f"{action}: {q_value:7.2f} {'←-- Best' if q_value == max(q_values) else ''}")
            print()




if __name__ == "__main__":
    # Train the agent
    print("Training agent...")
    visualize_q_learning_step()
    q_changes, avg_rewards, episode_lengths = train_agent()
    
    # Plot training results
    plot_training_results(q_changes, avg_rewards, episode_lengths)

    # Example output after training
    #visualize_q_values()

    # Run the demonstration
    #demonstrate_value_gradient()
    
    # Test the learned policy
    optimal_path = test_agent()
    print("\nLearned Q-table:")
    print(Q)
    print("\nOptimal path found:", optimal_path) 