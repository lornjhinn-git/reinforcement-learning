from .. import sarsa
from ..constants import constants as Constantor
from ..data import preprocess as Preprocessor
import pandas as pd 
import numpy as np
from sqlalchemy import create_engine
from functools import partial
import math
import random

global Sarsa
Sarsa = sarsa.SARSA()

# Generate the sample data
def train_test_split(df) -> tuple[pd.DataFrame, pd.DataFrame]:

    train_data = df[:math.floor(df.shape[0]*Constantor.TRAIN_SPLIT)]
    test_data = df[train_data.shape[0]:]

    return train_data, test_data


def policy(state, verbose = False) -> tuple[int, float]: 
	best_action = None
	best_value = float('-inf')
	global Sarsa
	
	# update allowed actions everytime based on agent current holding unit 
	if Sarsa.isHolding == False: # indicate can buy/no action but cannot sell
		allowed_actions = ['buy', 'no_action']
	else:
		allowed_actions = ['sell', 'no_action']

	random.shuffle(allowed_actions)

	for action in allowed_actions:
		if verbose:
			print(f"Holding: {Sarsa.isHolding}")
			print(f'action: {action}')
			print(f'value: {Sarsa.Q[state][Constantor.ACTION_DICT.get(action)]} vs best_value: {best_value}')
			print(f'new best action: {action}')
		if Sarsa.Q[state][Constantor.ACTION_DICT.get(action)] > best_value:
			best_action = Constantor.ACTION_DICT.get(action)
			best_value = Sarsa.Q[state][best_action]
				
	r_var = random.random()
	if r_var < Sarsa.epsilon:
		if verbose:
			print(f'Choosing random action')
		best_action = Constantor.ACTION_DICT.get(random.choice(allowed_actions))
		best_value = Sarsa.Q[state][best_action]
		
	if verbose:
		print(f'Final action: {best_action}\n')

	return best_action, best_value

# Update Q-value for a state-action pair based on observed rewards and estimated future Q-values
def update_q_value(state:tuple, action:int, rewards:list, rewards_value:float, next_state:tuple, next_action:int, verbose=False):

	if verbose:
		print(f"State: {state}, Action: {action}, Rewards: {rewards}, Next_state: {next_state}, Next_action: {next_action}")
		
	# Compute the updated Q-value using the SARSA update equation
	current_q = Sarsa.Q[state][Constantor.ACTION_DICT.get(action)]

	# Additional reward if have been making profit of at least 20 usd
	if sum(rewards) >= 20: current_q += 100
	next_q = Sarsa.Q[next_state][Constantor.ACTION_DICT.get(next_action)]
	new_q = current_q + Sarsa.learning_rate * (rewards_value + Sarsa.gamma * next_q - current_q)
    
    # Update the Q-value in the Q-table
	Q[state][Constantor.ACTION_DICT.get(action)] = new_q
	
    # Check if the (state, action) pair exists in the Q-table
    # if (state, action) not in Q:
    #     Q[(state, action)] = 0.0


def preprocess(df, verbose=True) -> pd.DataFrame:
	df_preprocessed = df.pipe(Preprocessor.get_day_of_week)\
						.pipe(Preprocessor.set_action)\
						.pipe(Preprocessor.pareto_distribution_bins)\
						.pipe(Preprocessor.encode_time)
	
	df_preprocessed_volumes = df_preprocessed[Constantor.DESIRED_COL]
	df_preprocessed_volumes.columns = Constantor.RENAMED_COL
	df_preprocessed_volumes = df_preprocessed_volumes.pipe(Preprocessor.convert_to_first_day_of_month)\
	 												 .pipe(Preprocessor.get_week_of_month)

	df_preprocessed_prices = Preprocessor.get_daily_average_trade_total_price(df_preprocessed_volumes)

	df_volume_stats = df_preprocessed_volumes[Constantor.VOLUME_COLS].groupby(Constantor.GROUPBY_KEYS).describe().reset_index()
	df_price_stats = df_preprocessed_volumes[Constantor.PRICE_COLS].groupby(Constantor.GROUPBY_KEYS).mean()
	df_reward_stats = df_preprocessed_volumes[Constantor.REWARD_COLS[:-1]].groupby(Constantor.GROUPBY_KEYS).describe().reset_index()

	if verbose:
		print("\nPreprocessed dataframe:", df_preprocessed.head(3))
		print("\nPreprocessed volume dataframe:", df_preprocessed_volumes.head(3))
		print("\nPreprocessed price dataframe:", df_preprocessed_prices.head(3))
		print("\nVolume stats:", df_volume_stats)
		print("\nPrice stats:", df_price_stats)
		print("\nReward stats:", df_reward_stats)

	# generate reward table & initialize Q table
	reward_table, Q = Preprocessor.create_reward_table(df_reward_stats)
	
	Sarsa.reward_table = reward_table
	Sarsa.Q = Q

	return df


def train(data: np.array, verbose=True) -> dict:
	environments_list = []
	total_rewards_list = []
	rewards_list = []
	steps_list = []

	for episode in range(Sarsa.num_episodes):

		print("\nEpisode:", episode)

		# initialize cumulative rewards
		total_rewards = 0
		steps = []
		environments = []
		rewards = []

		current_state = (0,0,0) # Starting state
		action, action_value = policy(current_state)
		# update upcoming allowed actions
		if action == 0:
			Sarsa.isHolding = True
		else:
			Sarsa.isHolding = False

		rewards_value = Sarsa.Sarsa.reward_table[current_state][action]
		total_rewards += rewards_value

		steps.append(action)
		rewards.append(rewards_value)
		environments.append(current_state)

		# when current state has not iterate until the last row of Q table
		# while (current_state != (Sarsa.reward_table.shape[0],Sarsa.reward_table.shape[1],Sarsa.reward_table.shape[2])):

		for month in range(Sarsa.reward_table.shape[0]):
			for day in range(Sarsa.reward_table.shape[1]):
				for time in range(Sarsa.reward_table.shape[2]):

					# when iterating (0,0) start from (0,0,1) to (0,0,287) because (0,0,0) already initialized on top
					if current_state[0] == 0 and current_state[1] == 0:
						current_state = [month, day, time + 1]
					else:
						current_state = [month, day, time]

					current_state = tuple(current_state)
					# print("Current state:", current_state)
					# print("Current action:", action, action_value)

					rewards_value = Sarsa.reward_table[current_state][action]
					total_rewards += rewards_value

					# print("Total rewards:", total_rewards)

					steps.append(action)
					rewards.append(rewards_value)
					environments.append(current_state)

					# if not last row of state then + 1 else move up 1 level then + 1
					if current_state[2] < Sarsa.reward_table.shape[2] - 1:
						next_state = [current_state[0], current_state[1], current_state[2] + 1]
						# print("In time:", current_state, next_state)
					elif Sarsa.reward_table.shape[2] - 1 == current_state[2] and current_state[1] < Sarsa.reward_table.shape[1] - 1:
						next_state = [current_state[0], current_state[1] + 1, current_state[2]]
						# print("In day:", current_state,  next_state)
					elif current_state[0] < Sarsa.reward_table.shape[0] - 1:
						next_state = [current_state[0] + 1, current_state[1], current_state[2]]
						# print("In month:", current_state, next_state)
					
					next_state = tuple(next_state)
					next_action, next_action_value = policy(next_state)

					#print("End:", current_state, action, rewards, next_state, next_action)
					update_q_value(current_state, action, rewards, rewards_value, next_state, next_action, verbose=False)

					# print(f'After update:, action: {action}, action_value : {action_value}, next_action: {next_action}, next_action_value: {next_action_value}\n')
					# update upcoming allowed actions
					if next_action == 0:
						Sarsa.isHolding = True
					if next_action == 1:
						Sarsa.isHolding = False

					current_state = next_state
					action = next_action 
					action_value = next_action_value

		total_rewards_list.append(sum(rewards))
		rewards_list.append(rewards)
		steps_list.append(steps)
		environments_list.append(environments)


def main(verbose=True):
	SARSA = sarsa.SARSA()
	engine = create_engine(f'postgresql://postgres:postgres@localhost:5432/{Constantor.DATABASE_NAME}')

	df = pd.read_sql_query(f'select * from {Constantor.TABLE_NAME}',con=engine)\
			   .query(f"period_type == '{Constantor.PERIOD_TYPE}'").reset_index()
	
	# Split data for train test
	df_train, df_test = train_test_split(df)
	df_preprocessed_train = preprocess(df_train)
	train(df_preprocessed_train)
	


if __name__ == '__main__':
	print("Start training process")
	main()


