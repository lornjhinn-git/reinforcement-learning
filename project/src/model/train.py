from .. import sarsa
from ..constants import constants as Constantor
from ..data import preprocess as Preprocessor, utils as Utilator
import pandas as pd 
import numpy as np
from sqlalchemy import create_engine
from functools import partial
from datetime import datetime
import math
import random
import joblib

global Sarsa
Sarsa = sarsa.SARSA()

# Generate the sample data
def train_test_split(df):
    global Sarsa
    Sarsa.train_data = df[:math.floor(df.shape[0]*Constantor.TRAIN_SPLIT)]
    Sarsa.test_data = df[math.floor(df.shape[0]*Constantor.TRAIN_SPLIT):]
    ##############
	# train data #
	##############
    Sarsa.train_data.to_csv("train_data.csv")
    Sarsa.train_data.to_csv(f"train_data_{Utilator.get_formatted_date()}.csv")
    ##############
	# test data  #
	##############
    Sarsa.test_data.to_csv("test_data.csv")
    Sarsa.test_data.to_csv(f"test_data_{Utilator.get_formatted_date()}.csv")
    print(f"Total sample size: {df.shape}")
    print(f"Sarsa train data sample size: {Sarsa.train_data.shape}")
    print(f"Sarsa test data sample size: {Sarsa.test_data.shape}")


def policy(state, verbose = False) -> tuple[int, float]: 
	best_action = None
	best_value = float('-inf')
	global Sarsa
	
	# update allowed actions everytime based on agent current holding unit 
	if state[0] == 1: # if in holding state, then only can sell or no action
		allowed_actions = ['sell', 'no_action']
	else:
		allowed_actions = ['buy', 'no_action']

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
	Sarsa.Q[state][Constantor.ACTION_DICT.get(action)] = new_q
	
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


def train(data: np.array, verbose=True) -> tuple[np.array, dict]:
	environments_list = []
	total_rewards_list = []
	rewards_list = []
	steps_list = []

	global Sarsa

	for episode in range(Sarsa.num_train_episodes):

		print("\nEpisode:", episode)

		# initialize cumulative rewards
		total_rewards = 0
		steps = []
		environments = []
		rewards = []

		# # Randomize the holding status to determine the starting state
		# Sarsa.isHolding = random.random() < 0.5
		# if Sarsa.isHolding == True:
		# 	current_state = (1,0,0,0) # Starting state
		# else:
		# 	current_state = (0,0,0,0)
		# action, action_value = policy(current_state)

		# # update upcoming allowed actions and new state
		# if action == 0:
		# 	Sarsa.isHolding = True
		# 	# current_state = (1,) + current_state[1:]
		# else:
		# 	Sarsa.isHolding = False
		# 	# current_state = (0,) + current_state[1:]

		# rewards_value = Sarsa.reward_table[current_state][action]
		# total_rewards += rewards_value

		# Randomize the starting state to be holding or not holding
		if random.random() < 0.5: 
			current_state = (0,0,0,0)
		else:
			current_state = (1,0,0,0)

		# the recommended action will be choosen from the allowed actions listS
		action, action_value = policy(current_state)

		# get the reward from the action taken in current state
		rewards_value = Sarsa.reward_table[current_state][action]
		total_rewards += rewards_value

		steps.append(action)
		rewards.append(rewards_value)
		environments.append(current_state)

		# when current state has not iterate until the last row of Q table
		# while (current_state != (Sarsa.reward_table.shape[0],Sarsa.reward_table.shape[1],Sarsa.reward_table.shape[2])):

		# dummy initialization for next state
		next_state = current_state

		for month in range(0, Sarsa.reward_table.shape[1]):
			for day in range(0, Sarsa.reward_table.shape[2]):
				for time in range(0, Sarsa.reward_table.shape[3]):

					# only modify the next state to start on 2nd state when the incoming state 
					# is (0,0,0,0) or (1,0,0,0)
					if current_state[1:] == (0,0,0):
						next_state = list(next_state)
						next_state[3] = 1
						next_state = tuple(next_state) # add 1 to time state if starting from (1,0,0,0) or (0,0,0,0)
					else:
						next_state = tuple([next_state[0], month, day, time])

					# second layer modification on state based on the current action
					if action == 0: # buy, next state become holding state
						next_state = (1,) + (month, day, time)
					elif action == 1: # sell, next state become not holding state 
						next_state = (0,) + (month, day, time)
					else:  # if no action, then stick to the current holding state
						# print("No action")
						pass

					# current_state = tuple(current_state)
					# action, action_value = policy(current_state)
					# rewards_value = Sarsa.reward_table[current_state][action]
					# total_rewards += rewards_value

					# steps.append(action)
					# rewards.append(rewards_value)
					# environments.append(current_state)

					# if not last row of state then + 1 else move up 1 level then + 1
					# if current_state[3] < Sarsa.reward_table.shape[3] - 1:
					# 	next_state = [current_state[0], current_state[1], current_state[2], current_state[3] + 1]
					# 	if verbose:
					# 		print("In time:", current_state, next_state)
					# elif Sarsa.reward_table.shape[3] - 1 == current_state[3] and current_state[2] < Sarsa.reward_table.shape[1] - 1:
					# 	next_state = [current_state[0], current_state[1], current_state[2] + 1, current_state[3]]
					# 	if verbose:
					# 		print("In day:", current_state,  next_state)
					# elif current_state[1] < Sarsa.reward_table.shape[1] - 1:
					# 	next_state = [current_state[0], current_state[1] + 1, current_state[2], current_state[3]]
					# 	if verbose:
					# 		print("In month:", current_state, next_state)
					
					# next_state = tuple(next_state)
					next_action, next_action_value = policy(next_state)

					#print("End:", current_state, action, rewards, next_state, next_action)
					update_q_value(current_state, action, rewards, rewards_value, next_state, next_action, verbose=False)

					# print(f'After update:, action: {action}, action_value : {action_value}, next_action: {next_action}, next_action_value: {next_action_value}\n')
					# update upcoming allowed actions
					# if next_action == 0 and Sarsa.isHolding == False: # want to buy and is allow to buy
					# 	Sarsa.isHolding = True
					# 	next_state = (1,) + next_state[1:]
					# if next_action == 1 and Sarsa.isHolding == True: # want to sell and is allow to sell
					# 	Sarsa.isHolding = False
					# 	next_state = (0,) + next_state[1:]

					if verbose:
						print("\nCurrent state:", current_state)
						print("Current action:", action)
						print("Next state:", next_state)
						print("Next action:", next_action)

					current_state = next_state
					action = next_action 
					action_value = next_action_value

		total_rewards_list.append(sum(rewards))
		rewards_list.append(rewards)
		steps_list.append(steps)
		environments_list.append(environments)
	
	Sarsa.train_value_dict['environments'] = environments_list 
	Sarsa.train_value_dict['total_rewards'] = total_rewards_list
	Sarsa.train_value_dict['rewards'] = rewards_list
	Sarsa.train_value_dict['steps'] = steps_list

	print("Finish training. All values are stored in Sarsa!")


def save_model():	
	joblib.dump(Sarsa.Q, f'sarsa_crypto.joblib')
	joblib.dump(Sarsa.Q, f'sarsa_crypto_{Utilator.get_formatted_date()}.joblib')

	for key in Sarsa.train_value_dict:
		with open(f'{key}.txt', 'w') as file:
			for item in Sarsa.train_value_dict.get(key):
				file.write(str(item) + '\n')

	print("Finish saving model and values dictionary!")


def validate():
	if Sarsa.Q is None:
		# read in the saved model 
		Sarsa.Q = joblib.load('sarsa_crypto.joblib')
		print("Loaded in Sarsa model!")
	
	if Sarsa.test_data is None: 
		Sarsa.test_data = pd.read_csv("test_data.csv")
		print("Loaded in test data!")

	df_preprocessed_test = preprocess(Sarsa.test_data)

	environments_list = []
	total_rewards_list = []
	rewards_list = []
	steps_list = []

	for episode in range(Sarsa.num_test_episodes):

		total_rewards = 0
		steps = []
		environments = []
		rewards = []

		# hard assign the starting holding state, can determine by ourselves to start with holding or not holding
		# currently set to not holding as dont want to hold overnight while sleeping and not monitoring 
		holding_state = 0

		# 28/07/2023: currently the validation will always start from 1/0 + (0,0,0), which mean always starting from a new day new time
		# in the future can add in the enhanced version where it start from where the train data last 
		# Note: the reward table here is the validation reward table 
		current_state = (holding_state,0,0,0)
		action = np.argmax(Sarsa.Q[current_state])
		rewards_value = Sarsa.reward_table[current_state][action]

		steps.append(action)
		rewards.append(rewards_value)
		environments.append(current_state)

		# dummy initialization for next state
		next_state = current_state

		for month in range(0, Sarsa.reward_table.shape[1]):
			for day in range(0, Sarsa.reward_table.shape[2]):
				for time in range(0, Sarsa.reward_table.shape[3]):
					if current_state[1:] == (0,0,0):
						next_state = list(next_state)
						next_state[3] = 1
						next_state = tuple(next_state) # add 1 to time state if starting from (1,0,0,0) or (0,0,0,0)
					else:
						next_state = tuple([next_state[0], month, day, time])
				
					# second layer modification on state based on the current action
					if action == 0: # buy, next state become holding state
						next_state = (1,) + (month, day, time)
					elif action == 1: # sell, next state become not holding state 
						next_state = (0,) + (month, day, time)
					else:  # if no action, then stick to the current holding state
						# print("No action")
						pass

					next_action = np.argmax(Sarsa.Q[next_state])

					current_state = next_state
					action = next_action
					rewards_value = Sarsa.reward_table[current_state][action]

					steps.append(action)
					rewards.append(rewards_value)
					environments.append(current_state)

		total_rewards += sum(rewards)
		Sarsa.test_value_dict['environments'] = environments_list 
		Sarsa.test_value_dict['total_rewards'] = total_rewards_list
		Sarsa.test_value_dict['rewards'] = rewards_list
		Sarsa.test_value_dict['steps'] = steps_list

	average_rewards = total_rewards / Sarsa.num_test_episodes
	print(f"Average reward on test data: {average_rewards}") 


def main(isTraining=True, verbose=True):
	SARSA = sarsa.SARSA()
	engine = create_engine(f'postgresql://postgres:postgres@localhost:5432/{Constantor.DATABASE_NAME}')

	df = pd.read_sql_query(f'select * from {Constantor.TABLE_NAME}',con=engine)\
			   .query(f"period_type == '{Constantor.PERIOD_TYPE}'").reset_index()
	
	# Split data for train test
	if isTraining:
		train_test_split(df)
		df_preprocessed_train = preprocess(Sarsa.train_data)
		train(df_preprocessed_train, verbose=verbose)
		save_model()

	validate()


if __name__ == '__main__':
	print("Start training process")
	main(isTraining=True, verbose=False)


