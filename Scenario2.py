import random
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from FourRooms import FourRooms
import sys

# -----------------------------
# Constants for Q-learning
# -----------------------------
EPISODES = 3000               # More episodes due to increased complexity (4 packages)
ALPHA = 0.1                   # Learning rate (Q-value adjustment factor)
GAMMA = 0.9                   # Discount factor for future rewards
EPSILON_START = 1.0           # Initial epsilon for exploration
EPSILON_END = 0.05            # Minimum allowed epsilon
EPSILON_DECAY_RATE = 0.995    # Decay rate for epsilon per episode

# -----------------------------
# Action and grid mappings
# -----------------------------
actions = [FourRooms.UP, FourRooms.DOWN, FourRooms.LEFT, FourRooms.RIGHT]
action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
grid_names = ['EMPTY', 'RED', 'GREEN', 'BLUE']  # Used for logging grid types

# -----------------------------
# Smoothing function for reward curves
# -----------------------------
def smooth(data, window=20):
    box = np.ones(window) / window
    return np.convolve(data, box, mode='same')

# -----------------------------
# Training function for Q-learning
# -----------------------------
def train(stochastic=False):
    # Initialize environment in 'multi' mode (4 packages)
    env = FourRooms('multi', stochastic=stochastic)

    # Initialize Q-table with default zero values
    Q = defaultdict(lambda: [0, 0, 0, 0])
    rewards = []
    epsilon = EPSILON_START

    for episode in range(EPISODES):
        env.newEpoch()  # Start new training episode
        state = (*env.getPosition(), env.getPackagesRemaining())
        total_reward = 0

        while not env.isTerminal():
            # Epsilon-greedy policy: explore or exploit
            if random.random() < epsilon:
                action = random.choice(range(4))  # Random exploration
            else:
                action = max(range(4), key=lambda a: Q[state][a])  # Greedy action

            # Take action and observe result
            grid_type, new_pos, packages_left, is_terminal = env.takeAction(actions[action])

            # Assign reward: +100 for collecting a package, -1 for movement
            reward = 100 if grid_type > 0 else -1
            total_reward += reward

            # Q-learning update rule
            new_state = (*new_pos, packages_left)
            best_next = max(Q[new_state])
            Q[state][action] += ALPHA * (reward + GAMMA * best_next - Q[state][action])

            state = new_state  # Move to new state

        rewards.append(total_reward)  # Track reward per episode
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY_RATE)  # Decay epsilon

    # -----------------------------
    # Final evaluation (greedy-only)
    # -----------------------------
    env.newEpoch()
    state = (*env.getPosition(), env.getPackagesRemaining())
    print("\n============================================")
    print("             FINAL PATH TRACE")
    print("============================================")
    print(f"[Start] Agent begins at {state[:2]} with {state[2]} package(s) left")

    step = 1
    while not env.isTerminal():
        action = max(range(4), key=lambda a: Q[state][a])
        grid_type, new_pos, packages_left, is_terminal = env.takeAction(actions[action])
        print(f"[Step {step}] Action: {action_names[action]} -> {new_pos} | Grid: {grid_names[grid_type]} | Packages left: {packages_left}")
        if grid_type > 0:
            print("Package collected!")
        if is_terminal:
            print("Reached terminal state")
        state = (*new_pos, packages_left)
        step += 1

    # Save final path visualization to PNG
    suffix = 'stochastic' if stochastic else 'deterministic'
    env.showPath(-1, savefig=f'final_path_multi_{suffix}.png')

    return rewards

# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    # Check for optional stochastic argument
    stochastic = '--stochastic' in sys.argv

    # Train the agent
    rewards = train(stochastic=stochastic)

    # -----------------------------
    # Plot learning curve
    # -----------------------------
    plt.figure()
    plt.plot(rewards, label='Raw Reward', alpha=0.4)
    plt.plot(smooth(rewards), label='Smoothed Reward', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    title = 'Scenario 2 - Multiple Package Collection'
    if stochastic:
        title += ' (Stochastic)'
    plt.title(title)
    plt.legend()
    plt.savefig(f'learning_curve_multi_{"stochastic" if stochastic else "deterministic"}.png')
    plt.show()
