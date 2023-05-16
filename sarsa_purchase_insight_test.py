import random
import numpy as np

# Define the state space
num_segments = 2
num_days = 7
num_pages = 4
state_space = np.zeros((num_segments, num_days, num_pages))


# Define the action space
num_prices = 10
num_times = 20
action_space = np.zeros((num_times, num_prices))


# Initialize the Q-values
Q = np.zeros((num_segments, num_days, num_pages, num_times, num_prices))


# Define the reward function
def reward(price, cost):
    return price - cost


# Define a function to get the current state
def get_state(num_segments, num_days, num_pages, num_times, num_prices):
    # concatenate scalar values into a numpy array
    state = np.array([num_segments, num_days, num_pages, num_times, num_prices])
    return state


def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        # Randomly choose an action
        action = np.random.choice((0, num_prices-1))
    else:
        # Choose the action with highest Q value
        action = np.argmax(Q[state[0]][state[1]][state[2]][state[3]][state[4]])
    return action


def train_sarsa(train_data):

    # Define variables to track performance validation
    prev_avg_reward = 0
    avg_reward = 0
    num_episodes = 0
    
    # Train for multiple episodes until convergence
    while abs(avg_reward - prev_avg_reward) > 0.001:
        prev_avg_reward = avg_reward
        total_reward = 0
        print("Start SARSA")
        
        # Loop through all training data
        for i in range(len(train_data)-1):
            # Get current state and action
            state = get_state(train_data[i][0], train_data[i][1], train_data[i][2], train_data[i][3], train_data[i][4])
            action = choose_action(state)
            
            # Get next state and reward
            next_state = get_state(train_data[i+1][0], train_data[i+1][1], train_data[i+1][2], train_data[i+1][3], train_data[i+1][4])
            reward = train_data[i+1][5] * 0.1
            
            # Choose next action based on epsilon-greedy policy
            next_action = choose_action(next_state)
            
            # Update Q table
            td_error = reward + gamma * Q[next_state[0], next_state[1], next_state[2], next_state[3], next_state[4]] - Q[state[0], state[1], state[2], state[3], state[4]]
            Q[state[0], state[1], state[2], state[3], state[4]] += alpha * td_error
            
            total_reward += reward
        
        # Calculate average reward for the current episode
        avg_reward = total_reward / len(train_data)
        num_episodes += 1
        print("Episode:", num_episodes, "Average Reward:", avg_reward)
    
    return Q


# Define the SARSA algorithm
def sarsa(state, action, reward, next_state, next_action, alpha, gamma, epsilon):
    # Get the current Q-value
    q = Q[state[0], state[1], state[2], action[0], action[1]]
    
    # Get the next Q-value
    next_q = Q[next_state[0], next_state[1], next_state[2], next_action[0], next_action[1]]
    
    # Update the Q-value using SARSA
    delta = reward + gamma * next_q - q
    Q[state[0], state[1], state[2], action[0], action[1]] += alpha * delta
    
    # Choose the next action using epsilon-greedy policy
    if np.random.rand() < epsilon:
        next_action = np.random.randint(num_times), np.random.randint(num_prices)
    else:
        next_action = np.unravel_index(np.argmax(Q[next_state[0], next_state[1], next_state[2]]), (num_times, num_prices))
    
    return next_action


# Generate the sample data
def generate_sample():
    num_transactions = 1000
    dataset = []
    testset = []

    for i in range(num_transactions):
        # Generate a random state
        segment = np.random.randint(num_segments)
        day = np.random.randint(num_days)
        page = np.random.randint(num_pages)
        state = (segment, day, page)
        
        # Choose an action using epsilon-greedy policy
        if np.random.rand() < 0.1:
            price = np.random.randint(num_prices)
            time = np.random.randint(num_times)
        else:
            time, price = np.unravel_index(np.argmax(Q[segment, day, page]), (num_times, num_prices))
        action = (time, price)
        
        # Generate a random cost
        cost = np.random.randint(10, 101)
        
        # Compute the reward
        reward = action[0] - cost
        
        # Add the data to the dataset or testset
        if i < num_transactions * 0.8:
            dataset.append([segment, day, page, time, price, cost])
        else:
            testset.append([segment, day, page, time, price, cost])

    return dataset, testset


dataset, testset = generate_sample()
# Train the SARSA algorithm on the data
data = [[segment, day, page, time, price, cost] for segment, day, page, time, price, cost in dataset]
alpha = 0.1
gamma = 0.9
epsilon = 0.1

train_sarsa(data)

# for i in range(len(data)-1):
#     state = tuple(data[i][:3])
#     action = tuple(data[i][3:5])
#     cost = data[i][5]
#     next_state = tuple(data[i+1][:3])
#     next_action = tuple(data[i+1][3:5])
#     r = reward(action[0], cost)
    
#     sarsa(state, action, r, next_state, next_action, alpha, gamma, epsilon)
    

# # Evaluate the SARSA algorithm on the test data
test_data = [[segment, day, page, time, price, cost] for segment, day, page, time, price, cost in testset]
for i in range(len(test_data)):
    # get the current state
    state = get_state(test_data[i][0], test_data[i][1], test_data[i][2], test_data[i][3], test_data[i][4])
    
    # get the best action from the SARSA model
    action = np.argmax(Q[state[0], state[1], state[2], state[3], state[4]])

    # print the action for the current state
    print("Action for state {}: {}".format(i+1, action))
    

# for i in range(len(test_data)-1):
#     state = tuple(test_data[i][:3])
#     action = tuple(test_data[i][3:5])
#     cost = test_data[i][5]
#     next_state = tuple(test_data[i+1][:3])
#     next_action = tuple(test_data[i+1][3:5])
#     r = reward(action[0], cost)
#     total_reward += r
    
#     sarsa(state, action, r, next_state, next_action, alpha, gamma, epsilon)

# print("Total reward: ", total_reward)
