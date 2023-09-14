import gym
import numpy as np

# Create the Cliff Walking environment
env = gym.make('CliffWalking-v0')

# Reset the environment to its initial state
observation = env.reset()

# Set the number of steps to take
num_steps = 10
def epsilon_greedy_policy(state, q_table, epsilon=0.1):
    """
    Epsilon greedy policy implementation takes the current state and q_value table
    Determines which action to take based on the epsilon-greedy policy

    Args:
        epsilon -- type(float) Determines exploration/exploitation ratio
        state -- type(int) Current state of the agent value between [0:47]
        q_table -- type(np.array) Determines state value

    Returns:
        action -- type(int) Choosen function based on Q(s, a) pairs & epsilon
    """
    # choose a random int from an uniform distribution [0.0, 1.0)
    decide_explore_exploit = np.random.random()

    if (decide_explore_exploit < epsilon):
        action = np.random.choice(4)  # UP = 0, LEFT = 1, RIGHT = 2, DOWN = 3
    else:
        action = np.argmax(q_table[:, state])  # Choose the action with largest Q-value (state value)
        # print("argmax action: ", action)

    return action
# Take the given number of steps
for i in range(num_steps):
    # Render the environment to the screen
    env.render()

    # Choose a random action
    action = env.action_space.sample()

    # Take the action and get the next observation, reward, and done flag
    o, r, done, info, _ = env.step(action)

    # Print some environmental values
    print(f'Step {i}: observation={observation}, \
          reward={r}, done={done}, info={info}')

    # If the episode is over, reset the environment
    if done:
        observation = env.reset()

# Close the environment
env.close()