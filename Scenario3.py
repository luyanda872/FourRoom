import random
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from FourRooms import FourRooms
import sys

# ---------------------------------------
# Q-Learning Hyperparameters
# ---------------------------------------
EPISODES = 5000  # More training episodes due to strict order constraint (R → G → B)
ALPHA = 0.1      # Learning rate
GAMMA = 0.9      # Discount factor
EPSILON_START = 1.0   # Initial exploration rate
EPSILON_END = 0.05    # Minimum exploration rate
EPSILON_DECAY_RATE = 0.995  # How fast epsilon decays per episode

# ---------------------------------------
# Actions and Environment Constants
# ---------------------------------------
actions = [FourRooms.UP, FourRooms.DOWN, FourRooms.LEFT, FourRooms.RIGHT]
action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
grid_names = ['EMPTY', 'RED', 'GREEN', 'BLUE']

# ---------------------------------------
# Smoothing Function for Reward Plot
# ---------------------------------------
def smooth(data, window=20):
    """Apply moving average to reward data for smoother plots."""
    box = np.ones(window) / window
    return np.convolve(data, box, mode='same')

# ---------------------------------------
# Q-Learning Training Loop
# ---------------------------------------
def train(stochastic=False):
    # Initialize environment for RGB scenario (ordered package collection)
    env = FourRooms('rgb', stochastic=stochastic)
    Q = defaultdict(lambda: [0, 0, 0, 0])  # Q-table mapping state -> action values
    rewards = []
    epsilon = EPSILON_START

    for episode in range(EPISODES):
        env.newEpoch()
        state = (*env.getPosition(), env.getPackagesRemaining())  # (x, y, packages_left)
        total_reward = 0

        while not env.isTerminal():
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice(range(4))
            else:
                action = max(range(4), key=lambda a: Q[state][a])

            # Take action and observe transition
            grid_type, new_pos, packages_left, is_terminal = env.takeAction(actions[action])

            # Reward design:
            if is_terminal and packages_left > 0:
                reward = -100  # Penalty for collecting out of order
            elif grid_type > 0:
                reward = 100   # Correct package collected
            else:
                reward = -1    # Step cost

            total_reward += reward

            # Q-learning update rule
            new_state = (*new_pos, packages_left)
            best_next = max(Q[new_state])
            Q[state][action] += ALPHA * (reward + GAMMA * best_next - Q[state][action])
            state = new_state

        rewards.append(total_reward)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY_RATE)  # Decay epsilon

    # ---------------------------------------
    # Final Evaluation (Greedy Policy Only)
    # ---------------------------------------
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

    # Save final path visualization
    suffix = 'stochastic' if stochastic else 'deterministic'
    env.showPath(-1, savefig=f'final_path_rgb_{suffix}.png')

    return rewards

# ---------------------------------------
# Main Script Entry Point
# ---------------------------------------
if __name__ == "__main__":
    # Allow stochastic flag via command-line argument
    stochastic = '--stochastic' in sys.argv

    # Train agent
    rewards = train(stochastic=stochastic)

    # Plot raw and smoothed reward curves
    plt.figure()
    plt.plot(rewards, label='Raw Reward', alpha=0.4)
    plt.plot(smooth(rewards), label='Smoothed Reward', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    title = 'Scenario 3 - Ordered Package Collection'
    if stochastic:
        title += ' (Stochastic)'
    plt.title(title)
    plt.legend()
    plt.savefig(f'learning_curve_rgb_{"stochastic" if stochastic else "deterministic"}.png')
    plt.show()
