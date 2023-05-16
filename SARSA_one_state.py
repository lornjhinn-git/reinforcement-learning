import numpy as np
import pandas as pd

# Define the SARSA agent class
class SARSAAgent:
    def __init__(self, df_train, learning_rate, discount_factor, epsilon=0.1):
        self.df_train = df_train
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = None
        self.num_states = df_train['encoded_time'].nunique()
        self.num_actions = 2

    def initialize_q_table(self, df):

        df_sell = df[['encoded_time', 'sell_rewards']].copy()
        df_sell['action'] = 'sell'
        df_buy = df[['encoded_time', 'buy_rewards']].copy()
        df_buy['action'] = 'buy'

        # Pivot the DataFrame
        df_sell = pd.pivot_table(df_sell, values='sell_rewards',
                                index='encoded_time', columns='action').reset_index()

        df_buy = pd.pivot_table(df_buy, values='buy_rewards',
                                index='encoded_time', columns='action').reset_index()

        reward_array = df_sell.merge(df_buy, on='encoded_time', how='inner')[['sell', 'buy']].values
        # Ensure the array has the desired shape (n_time, n_action) by padding with zeros if necessary
        if reward_array.shape != (self.num_states, self.num_actions):
            padded_array = np.zeros((self.num_states, self.num_actions))
            padded_array[:reward_array.shape[0], :reward_array.shape[1]] = reward_array
            reward_array = padded_array

        # Display the resulting array
        print(reward_array)

        self.q_table = reward_array


    def get_action(self, state):
        # Epsilon-greedy policy for action selection
        if np.random.uniform(0, 1) < self.epsilon:
            # Explore: Choose a random action
            action = np.random.randint(0, self.num_actions)
        else:
            # Exploit: Choose the action with the highest Q-value
            action = np.argmax(self.q_table[state])
        return action

    def update_q_table(self, state, action, reward, next_state, next_action):
        # SARSA update rule
        current_q = self.q_table[state][action]
        next_q = self.q_table[next_state][next_action]
        td_error = reward + self.discount_factor * next_q - current_q
        updated_q = current_q + self.learning_rate * td_error
        self.q_table[state][action] = updated_q


# Function to get the reward for a given state-action pair
def get_reward(state, action):
    # Sample reward function for crypto trading
    # Here, we assume a positive reward for "Buy" action and a negative reward for "Sell" action
    if action == 0:  # Buy
        return 1
    elif action == 1:  # Sell
        return -1

# Function to get the next state for a given state-action pair
def get_next_state(state, action):
    # Sample state transition function for crypto trading
    # Here, we assume a simple transition where the next state is determined by the current state and action
    next_state = (state + action) % num_states
    return next_state

# After training, you can use the learned Q-table for decision-making
def make_decision(state):
    action = np.argmax(agent.q_table[state])
    return action

# # Define the environment and training parameters
# num_states = 10
# num_actions = 2
# learning_rate = 0.1
# discount_factor = 0.9
# epsilon = 0.1
# num_episodes = 1000

# # Create the SARSA agent
# # agent = SARSAAgent(num_states, num_actions, learning_rate, discount_factor, epsilon)

# # # Training loop
# # for episode in range(num_episodes):
# #     state = 0  # Starting state
# #     action = agent.get_action(state)

# #     for _ in range(num_steps_per_episode):
# #         # Execute action and observe the reward and next state
# #         reward = get_reward(state, action)
# #         next_state = get_next_state(state, action)
# #         next_action = agent.get_action(next_state)

# #         # Update the Q-table
# #         agent.update_q_table(state, action, reward, next_state, next_action)

# #         # Move to the next state
# #         state = next_state
# #         action = next_action






# import numpy as np
# import pandas as pd

# # Example data
# data = pd.DataFrame({
#     'sell_price': [10, 12, 15, 14, 16],
#     'buy_price': [8, 10, 12, 13, 11],
#     'time': ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05']
# })

# def initialize_q_table(num_states, num_actions):
#     return np.zeros((num_states, num_actions))


# def get_state(observation):
#     # Extract relevant features from the observation
#     return observation[['sell_price', 'buy_price']].values


# def epsilon_greedy_policy(q_table, state, epsilon):
#     if np.random.uniform() < epsilon:
#         action = np.random.randint(q_table.shape[1])
#     else:
#         action = np.argmax(q_table[state])

#     return action


# def take_action(action):
#     # Implement your action-taking logic here based on the specific problem
#     # This could involve buying, selling, or taking no action
#     if action == 0:
#         print("Taking no action")
#     elif action == 1:
#         print("Buying")
#     elif action == 2:
#         print("Selling")


# def get_reward(action, next_state, next_observation):
#     # Implement your reward function logic here based on the specific problem
#     # You can use the action, next_state, and next_observation to calculate the reward
#     # For example, you can calculate the profit based on the current and future prices
#     current_sell_price = next_observation['sell_price'].values[0]
#     current_buy_price = next_observation['buy_price'].values[0]

#     reward = 0  # Initialize the reward

#     if action == 1:  # Buying action
#         reward = current_sell_price - current_buy_price

#     elif action == 2:  # Selling action
#         reward = current_buy_price - current_sell_price

#     return reward


# def train_sarsa(data, num_states, num_actions, num_episodes, alpha, gamma, epsilon):
#     q_table = initialize_q_table(num_states, num_actions)
#     max_cumulative_reward = -np.inf  # Track the maximum cumulative reward

#     for episode in range(num_episodes):
#         observation = data.sample()  # Sample a random observation from the data
#         state = get_state(observation)
#         action = epsilon_greedy_policy(q_table, state, epsilon)

#         done = False
#         cumulative_reward = 0  # Track the cumulative reward for the current episode

#         while not done:
#             take_action(action)

#             next_observation = data.sample()  # Sample the next observation from the data
#             next_state = get_state(next_observation)
#             next_action = epsilon_greedy_policy(q_table, next_state, epsilon)

#             reward = get_reward(action, next_state, next_observation)

#             # SARSA update
#             q_table[state, action] += alpha * (
#                     reward + gamma * q_table[next_state, next_action] - q_table[state, action]
#             )

#             state = next_state
#             action = next_action
#             cumulative_reward += reward  # Accumulate the reward

#             done = cumulative_reward > max_cumulative_reward

#         if cumulative_reward > max_cumulative_reward:
#             max_cumulative_reward = cumulative_reward

#     return q_table


# # Test the trained SARSA model
# def test_sarsa():
#     state = get_initial_state()
#     action = np.argmax(Q[state])

#     while True:
#         next_state = get_next_state(state, action)
#         next_action = np.argmax(Q[next_state])

#         # Your code to perform actions based on the current state goes here

#         state = next_state
#         action = next_action

#         # Check for termination
#         if state == MAX_STATE:  # Replace MAX_STATE with your desired termination state
#             break

# # Main function
# def main():
#     num_states = 10  # Replace with the actual number of states
#     num_actions = 3  # Replace with the actual number of actions

#     # Initialize the Q table
#     initialize_q_table(num_states, num_actions)

#     # Train the SARSA model
#     train_sarsa(num_episodes=100)

#     # Test the trained SARSA model
#     test_sarsa()

# if __name__ == "__main__":
#     main()
