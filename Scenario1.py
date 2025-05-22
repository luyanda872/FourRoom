import random
import matplotlib.pyplot as plt
from collections import defaultdict
from FourRooms import FourRooms
import sys

# Constants for Q-learning
EPISODES = 2000               # Total training episodes
ALPHA = 0.1                   # Learning rate
GAMMA = 0.9                   # Discount factor for future rewards
EPSILON_FIXED = 0.1           # Constant exploration rate
EPSILON_DECAY_START = 1.0     # Initial exploration rate for decaying strategy
EPSILON_DECAY_END = 0.05      # Minimum epsilon for decaying strategy
EPSILON_DECAY_RATE = 0.995    # Decay rate per episode

# Action index mapping to environment constants
actions = [FourRooms.UP, FourRooms.DOWN, FourRooms.LEFT, FourRooms.RIGHT]
action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
grid_names = ['EMPTY', 'RED', 'GREEN', 'BLUE']  # For rendering cell types

# Define exploration strategies
strategies = {
    'fixed': lambda eps, ep: EPSILON_FIXED,  # Always use the same epsilon
    'decay': lambda eps, ep: max(EPSILON_DECAY_END, eps * EPSILON_DECAY_RATE)  # Decay over time
}

def train(strategy_name, stochastic=False):
    # Initialize epsilon and environment
    epsilon = EPSILON_DECAY_START
    env = FourRooms('simple', stochastic=stochastic)  # Create environment

    # Initialize Q-table: keys are state tuples, values are 4-action arrays
    Q = defaultdict(lambda: [0, 0, 0, 0])
    rewards = []

    for episode in range(EPISODES):
        env.newEpoch()  # Reset environment for new episode
        state = (*env.getPosition(), env.getPackagesRemaining())  # Initial state: position + packages

        total_reward = 0

        while not env.isTerminal():
            # Epsilon-greedy policy: explore or exploit
            if random.random() < epsilon:
                action = random.choice(range(4))  # Random action
            else:
                action = max(range(4), key=lambda a: Q[state][a])  # Best known action

            # Execute action and get transition feedback
            grid_type, new_pos, packages_left, is_terminal = env.takeAction(actions[action])

            # Assign reward based on outcome
            reward = 100 if grid_type > 0 else -1
            total_reward += reward

            # Q-learning update
            new_state = (*new_pos, packages_left)
            best_next = max(Q[new_state])
            Q[state][action] += ALPHA * (reward + GAMMA * best_next - Q[state][action])

            state = new_state  # Move to next state

        # Track reward for plotting
        rewards.append(total_reward)
        # Update epsilon for next episode (if decay strategy)
        epsilon = strategies[strategy_name](epsilon, episode)

    # Final evaluation episode (greedy only)
    env.newEpoch()
    state = (*env.getPosition(), env.getPackagesRemaining())

    # Logging final path
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

    # Save path visualization
    suffix = f'{strategy_name}_stochastic' if stochastic else strategy_name
    env.showPath(-1, savefig=f'final_path_{suffix}.png')

    return rewards, env  # Return learning performance history

if __name__ == "__main__":
    # Check for stochastic flag in command line
    stochastic = '--stochastic' in sys.argv

    # Run both strategies
    rewards_fixed, _ = train('fixed', stochastic=stochastic)
    rewards_decay, _ = train('decay', stochastic=stochastic)

    # Plotting reward curves for both strategies
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(rewards_fixed, label='Fixed Epsilon', color='blue')
    ax1.set_title('Fixed Epsilon')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend()

    ax2.plot(rewards_decay, label='Decaying Epsilon', color='green')
    ax2.set_title('Decaying Epsilon')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Reward')
    ax2.legend()

    plt.suptitle('Learning Curves Comparison' + (' (Stochastic)' if stochastic else ''))
    plt.tight_layout()
    plt.savefig(f'learning_curves_both_{"stochastic" if stochastic else "deterministic"}.png')
    plt.show()
