import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numba import njit  # Accelerates Python functions

class MazeEnvironment:
    """
    Creates a grid world environment with:
    - Grid dimensions
    - Starting position
    - Terminal states and their rewards
    - Wall positions
    - Movement actions (N/S/E/W)
    """
    def __init__(self, dimensions, start_location, terminal_positions, terminal_payouts, obstacles):
        self.dimensions = dimensions  # (rows, columns)
        self.start_location = start_location
        self.terminal_positions = terminal_positions  # List of end states
        self.terminal_payouts = terminal_payouts  # Rewards for terminal states
        self.obstacles = obstacles  # Wall coordinates
        self.action_count = 4  # Actions: 0=Up, 1=Down, 2=Right, 3=Left

    def initialize(self):
        """Resets agent to starting position"""
        self.current_location = self.start_location
        return self.current_location

    def move(self, action):
        """Executes movement action, returns new position/reward/done status"""
        row, col = self.current_location
        max_rows, max_cols = self.dimensions
        # Direction modifiers: [Up, Down, Right, Left]
        row_changes = [-1, 1, 0, 0]
        col_changes = [0, 0, 1, -1]

        new_row = row + row_changes[action]
        new_col = col + col_changes[action]
        next_position = (new_row, new_col)

        # Check boundary/wall collision
        if (new_row < 0 or new_row >= max_rows or new_col < 0 or new_col >= max_cols) or (next_position in self.obstacles):
            return self.current_location, -1, False  # Penalize invalid moves

        # Terminal state check
        if next_position in self.terminal_positions:
            reward_value = self.terminal_payouts[next_position]
            terminal_reached = True
        else:
            reward_value = 0
            terminal_reached = False

        self.current_location = next_position
        return next_position, reward_value, terminal_reached

@njit  # Compiles function for faster execution
def choose_epsilon_action(q_table, position, epsilon_val):
    """Selects action using epsilon-greedy strategy"""
    if np.random.rand() < epsilon_val:  # Explore
        return np.random.randint(4)
    else:  # Exploit
        return np.argmax(q_table[position])

@njit
def choose_softmax_action(q_table, position, temperature):
    """Selects action using softmax strategy"""
    action_values = q_table[position]
    # Numerical stability adjustment
    scaled_values = temperature * (action_values - np.max(action_values))
    exp_vals = np.exp(scaled_values)
    probabilities = exp_vals / np.sum(exp_vals)
    
    # Probabilistic selection
    rand_val = np.random.rand()
    cumulative = 0.0
    for action_idx in range(4):
        cumulative += probabilities[action_idx]
        if rand_val < cumulative:
            return action_idx
    return 3  # Default fallback

def learn_q_values(environment, total_episodes, learning_rate, discount_factor, action_selector, selector_param, max_attempts=1000, random_seed=None):
    """
    Trains Q-learning agent:
    1. Initializes Q-table
    2. Runs episodes
    3. Updates Q-values using TD learning
    4. Tracks performance metrics
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    rows, cols = environment.dimensions
    # Initialize Q-table (rows x cols x actions)
    q_table = np.zeros((rows, cols, environment.action_count))
    step_records = []
    successful_episodes = []

    for episode_idx in range(total_episodes):
        position = environment.initialize()
        step_count = 0
        completed = False

        # Episode loop with step limit
        while not completed and step_count < max_attempts:
            chosen_action = action_selector(q_table, position, selector_param)
            next_position, reward_given, completed = environment.move(chosen_action)
            
            # Q-value update rule
            if completed:
                target_value = reward_given
            else:
                target_value = reward_given + discount_factor * np.max(q_table[next_position])
                
            # Temporal Difference update
            q_table[position][chosen_action] += learning_rate * (target_value - q_table[position][chosen_action])
            
            position = next_position
            step_count += 1

        step_records.append(step_count)
        goal_position = list(environment.terminal_payouts.keys())[0]
        if next_position == goal_position:
            successful_episodes.append(step_count)
        else:
            successful_episodes.append(None)

        # Progress reporting
        if (episode_idx + 1) % 1000 == 0:
            print(f"Episode {episode_idx + 1}/{total_episodes} completed")

    return q_table, step_records, successful_episodes

def visualize_values(environment, q_table, plot_title):
    """Displays state value heatmap"""
    rows, cols = environment.dimensions
    state_values = np.max(q_table, axis=2)  # V(s) = max_a Q(s,a)
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(state_values, annot=True, fmt=".1f", cmap="viridis", ax=ax, cbar=True)
    ax.set_title(plot_title)
    plt.tight_layout()
    plt.show()

def visualize_policy(environment, q_table, plot_title):
    """Displays policy with arrows and special markers"""
    rows, cols = environment.dimensions
    optimal_actions = np.argmax(q_table, axis=2)  # Best action per state
    fig, ax = plt.subplots(figsize=(8, 8))
    # Background values
    sns.heatmap(np.max(q_table, axis=2), annot=False, cmap="viridis", ax=ax, cbar=False, alpha=0.2)
    # Action symbols: [Up, Down, Right, Left]
    action_symbols = ['↑', '↓', '→', '←']  

    for r in range(rows):
        for c in range(cols):
            # Wall marker
            if (r, c) in environment.obstacles:
                ax.text(c + 0.5, r + 0.5, '■', ha='center', va='center', fontsize=20, color='black')
            # Terminal state marker
            elif (r, c) in environment.terminal_positions:
                reward_val = environment.terminal_payouts[(r, c)]
                marker = '★' if reward_val > 0 else '☠'  # Star for positive, skull for negative
                ax.text(c + 0.5, r + 0.5, marker, ha='center', va='center', fontsize=20)
            # Action direction
            else:
                action_taken = optimal_actions[r, c]
                ax.text(c + 0.5, r + 0.5, action_symbols[action_taken], ha='center', va='center', fontsize=15)

    ax.set_title(plot_title)
    plt.tight_layout()
    plt.show()

def execute_scenario_one():
    """Part 1: Trains agent in 5x5 maze with epsilon-greedy exploration"""
    dimensions = (5, 5)
    start_point = (0, 0)
    terminal_points = [(4, 4), (2, 4)]
    terminal_rewards = {(4, 4): 5, (2, 4): -5}  # Goal vs trap
    barrier_locations = [(0, 3), (1, 1), (2, 2), (4, 2)]
    env = MazeEnvironment(dimensions, start_point, terminal_points, terminal_rewards, barrier_locations)

    episodes = 100000
    learning_rate = 0.1
    seed_value = 42

    # Gamma comparison
    for gamma_val in [0.1, 0.5, 0.9]:
        q_result, _, _ = learn_q_values(env, episodes, learning_rate, gamma_val, choose_epsilon_action, 0.1, seed_value)
        visualize_values(env, q_result, f"Scenario 1: State Values (γ={gamma_val}, ε=0.1)")
        visualize_policy(env, q_result, f"Scenario 1: Policy (γ={gamma_val}, ε=0.1)")

    # Epsilon comparison
    plt.figure(figsize=(12, 6))
    for epsilon_val in [0.1, 0.3, 0.5]:
        _, _, goal_steps = learn_q_values(env, episodes, learning_rate, 0.9, choose_epsilon_action, epsilon_val, seed_value)
        valid_steps = [s for s in goal_steps if s is not None]
        episode_indices = [i for i, s in enumerate(goal_steps) if s is not None]
        if valid_steps:
            window_size = max(1, len(valid_steps) // 100)  # Dynamic smoothing
            smoothed_avg = np.convolve(valid_steps, np.ones(window_size)/window_size, mode='valid')
            plt.plot(episode_indices[window_size-1:], smoothed_avg, label=f'ε={epsilon_val}')

    plt.xlabel('Training Episode')
    plt.ylabel('Steps to Goal (Smoothed)')
    plt.title("Scenario 1: Goal Convergence (γ=0.9)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def execute_scenario_two():
    """Part 2: Trains agent in 5x11 maze with softmax exploration"""
    dimensions = (5, 11)
    start_point = (0, 0)
    terminal_points = [(3, 6), (4, 10)]
    terminal_rewards = {(3, 6): -5, (4, 10): 5}  # Trap vs goal
    barrier_locations = [(1, 2), (2, 2), (3, 2), (0,5), (1,5), (3,5), (4,5), (2, 8), (2, 9), (4, 8)]
    env = MazeEnvironment(dimensions, start_point, terminal_points, terminal_rewards, barrier_locations)

    episodes = 200000
    learning_rate = 0.1
    seed_value = 42

    # Gamma comparison
    for gamma_val in [0.1, 0.5, 0.9]:
        q_result, _, _ = learn_q_values(env, episodes, learning_rate, gamma_val, choose_softmax_action, 0.1, seed_value)
        visualize_values(env, q_result, f"Scenario 2: State Values (γ={gamma_val}, β=0.1)")
        visualize_policy(env, q_result, f"Scenario 2: Policy (γ={gamma_val}, β=0.1)")

    # Beta (temperature) comparison
    plt.figure(figsize=(12, 6))
    for beta_val in [0.1, 0.3, 0.5]:
        _, _, goal_steps = learn_q_values(env, episodes, learning_rate, 0.9, choose_softmax_action, beta_val, seed_value)
        valid_steps = [s for s in goal_steps if s is not None]
        episode_indices = [i for i, s in enumerate(goal_steps) if s is not None]
        if valid_steps:
            window_size = max(1, len(valid_steps) // 100)
            smoothed_avg = np.convolve(valid_steps, np.ones(window_size)/window_size, mode='valid')
            plt.plot(episode_indices[window_size-1:], smoothed_avg, label=f'β={beta_val}')

    plt.xlabel('Training Episode')
    plt.ylabel('Steps to Goal (Smoothed)')
    plt.title("Scenario 2: Goal Convergence (γ=0.9)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    execute_scenario_one()
    execute_scenario_two()
