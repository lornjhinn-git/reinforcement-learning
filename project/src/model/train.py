from .. import sarsa
import numpy as np

SARSA = sarsa.SARSA()

# Generate the sample data
def generate_sample(df):

    cols = ['local_numeric_day', 'encoded_time', 'volume_bins', 'actions']
    df = df[cols]

    # Get states as tuple
    state_tuple = create_state_tuple(df)

    # Initiate Q table 
    global Q_star 
    Q_star = np.zeros((state_tuple))

    train_data = df[:math.floor(df.shape[0]*train_split)].to_numpy()
    test_data = df[train_data.shape[0]:].to_numpy()

    return train_data, test_data


def policy(Q, sarsa_agent, state, epsilon = 0.1, verbose = False) -> (int, float): 
	best_action = None
	best_value = float('-inf')
	
	# update allowed actions everytime based on agent current holding unit 
	if sarsa_agent.isHolding == False: # indicate can buy/no action but cannot sell
		allowed_actions = ['buy', 'no_action']
	else:
		allowed_actions = ['sell', 'no_action']

	random.shuffle(allowed_actions)

	for action in allowed_actions:
		if verbose:
			print(f"Holding: {sarsa_agent.isHolding}")
			print(f'action: {action}')
			print(f'value: {Q[state][action_dict.get(action)]} vs best_value: {best_value}')
			print(f'new best action: {action}')
		if Q[state][action_dict.get(action)] > best_value:
			best_action = action_dict.get(action)
			best_value = Q[state][best_action]
				
	r_var = random.random()
	if r_var < epsilon:
		if verbose:
			print(f'Choosing random action')
		best_action = action_dict.get(random.choice(allowed_actions))
		best_value = Q[state][best_action]
		
	if verbose:
		print(f'Final action: {best_action}\n')

	return best_action, best_value


# Update Q-value for a state-action pair based on observed rewards and estimated future Q-values
def update_q_value(state:tuple, action:int, rewards:list, rewards_value:float, next_state:tuple, next_action:int, verbose=False):

	if verbose == True:
		print(f"State: {state}, Action: {action}, Rewards: {rewards}, Next_state: {next_state}, Next_action: {next_action}")
		
	# Compute the updated Q-value using the SARSA update equation
	current_q = Q[state][action_dict.get(action)]

	# Additional reward if have been making profit of at least 20 usd
	if sum(rewards) >= 20: current_q += 100
	next_q = Q[next_state][action_dict.get(next_action)]
	new_q = current_q + lr * (rewards_value + GAMMA * next_q - current_q)
    
    # Update the Q-value in the Q-table
	Q[state][action_dict.get(action)] = new_q
	
    # Check if the (state, action) pair exists in the Q-table
    # if (state, action) not in Q:
    #     Q[(state, action)] = 0.0