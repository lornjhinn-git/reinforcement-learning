
# features that need to be scaled

# 1. month (jan - dec)
# 2. day (mon - sat)
# 3. time (12 am - 12 pm every 5 min)
# 4. actions (buy, sell, no action)

from sqlalchemy import create_engine
import pandas as pd 
import numpy as np
import random
import math 


import SARSA_one_state as SARSA
from datetime import date 
# transform encoded time into scalar value for easier indexing
from sklearn.preprocessing import LabelEncoder
import joblib 


label_encoder = LabelEncoder()

num_states = df['encoded_time'].nunique()
num_actions = df['actions'].nunique()
lr = 0.005
discount_factor = 0.1
epsilon = 0.1

sarsa_agent = SARSA.SARSAAgent(
    df,
    learning_rate=lr,
    discount_factor=discount_factor,
    epsilon=epsilon
)

global Q 
sarsa_agent.initialize_q_table(df)
Q = np.zeros(reward_table.shape)
print(Q.shape)

# the trade volumes here originally is "amount", total usd price is "vol"
# will rename after retrieved from the dataframe
desired_col = [ 
    'date', 
    'time', 
    'local_numeric_day', 
    'amount',
    'vol',
    'sell_rewards', 
    'buy_rewards', 
    'sell_cumulative_rewards', 
    'buy_cumulative_rewards',
    'actions',
    'volume_bins',
    'encoded_time'
]

renamed_col = [ 
    'date', 
    'time', 
    'local_numeric_day', 
    'trade_volumes',
    'trade_total_price',
    'sell_rewards', 
    'buy_rewards', 
    'sell_cumulative_rewards', 
    'buy_cumulative_rewards',
    'actions',
    'volume_bins',
    'encoded_time'
]

action_dict = {
	'buy': 0,
	'sell': 1,
	'no_action': 2
}

def get_day_of_week(df):
    """
    Returns the name of the day of the week for the given day number (0-6)
    """
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df['local_numeric_day'] =  df['datetime'].apply(lambda x: (x.weekday()) % 7 + 1)
    df['local_day'] =  df['local_numeric_day'].apply(lambda x: days[x-1])
    return df 


def set_action(df, optimum_sell_rewards=15, optimum_buy_rewards=15):
    """
    Adds a new column called 'price_diff' to the given DataFrame,
    containing the difference between the current row's close price and
    the previous row's close price.
    """
    # Create a new column called 'prev_close' that contains the close price from the previous row
    df['prev_close'] = df['close'].shift(1)

    # Compute the difference between the current row's close price and the previous row's close price
    df['price_diff'] = df['close'] - df['prev_close']
    df['sell_rewards'] = df['price_diff'].shift(-1)
    df['buy_rewards'] = (df['price_diff'].shift(-1))*-1
    df['sell_cumulative_rewards'] = df['sell_rewards'].cumsum()
    df['buy_cumulative_rewards'] = df['buy_rewards'].cumsum()
    df['actions'] = -1 # default 0 = buy, 1 = sell, -1 = no action
    df.loc[df['buy_rewards'] >= 5, 'actions'] = 0
    df.loc[df['sell_rewards'] > 5 , 'actions'] = 1
    df.loc[df['actions'] == 1, 'one_time_reward'] = df['sell_rewards']
    df.loc[df['actions'] == 0, 'one_time_reward'] = df['buy_rewards']
    df.loc[df['actions'] == -1, 'one_time_reward'] = 0

    # Return the updated DataFrame
    return df


# normal distribution optimum bin
def get_optimal_normal_distribution_num_bins(df):
    """
    Estimates the optimal number of bins for the 'volume_trade' column
    of the given DataFrame using the Freedman-Diaconis rule, and returns
    the estimated number of bins.
    """
    # Compute the interquartile range of the 'volume_trade' column
    q1, q3 = np.percentile(df['volume_trade'], [25, 75])
    iqr = q3 - q1

    # Estimate the optimal bin width using the Freedman-Diaconis rule
    bin_width = 2 * iqr / np.cbrt(len(df))

    # Compute the estimated number of bins
    num_bins = int(np.ceil((df['volume_trade'].max() - df['volume_trade'].min()) / bin_width))

    # Return the estimated number of bins
    return num_bins


# power law optimum bin 
def get_optimal_pareto_distribution_num_bins(df):
    """
    Estimates the optimal number of bins for the 'volume_trade' column
    of the given DataFrame using the Sturges method for power law distributions,
    and returns the estimated number of bins.
    """
    # Compute the sample size and the maximum value of the 'volume_trade' column
    n = len(df['amount'])
    x_max = df['amount'].max()

    # Estimate the optimal number of bins using the Sturges method
    num_bins = int(np.ceil(np.log2(n) + np.log2(1 + x_max)))

    # Return the estimated number of bins
    return num_bins


def pareto_distribution_bins(df, num_bins):
    """Creates power law bins for the 'volume_trade' column of the given
    DataFrame using the qcut function, and returns the updated DataFrame.
    """
    # Compute the quantiles of the 'volume_trade' column using a power law distribution
    quantiles = pd.qcut(df['amount'], num_bins, labels=False, duplicates='drop')

    # Add a new column to the DataFrame with the bin labels
    df['volume_bins'] = quantiles

    # Return the updated DataFrame
    return df


def encode_time(df):
    """Encodes the time in the given DataFrame as a string representing the time
    in sequential order (hour-minute-second), and returns the updated DataFrame.
    """
    # Convert the 'time' column to a datetime object
    df['time'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date

    # Extract the hour, minute, and second from the 'time' column
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    df['second'] = df['time'].dt.second

    # Convert the hour, minute, and second to strings
    df['hour_str'] = df['hour'].astype(str).str.zfill(2)
    df['minute_str'] = df['minute'].astype(str).str.zfill(2)
    df['second_str'] = df['second'].astype(str).str.zfill(2)

    # Concatenate the hour, minute, and second strings into a single time string
    df['encoded_time'] = df['hour_str'] + '-' + df['minute_str'] + '-' + df['second_str']

    # Drop the original hour, minute, and second columns
    df = df.drop(['hour', 'minute', 'second', 'hour_str', 'minute_str', 'second_str'], axis=1)

    # Return the updated DataFrame with the encoded time string
    return df


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


# Create a custom aggregation function to fill in values based on conditions
def fill_values(column):
    if column[column > 0].empty:
        return None
    return column[column > 0].values[0]


def convert_to_first_day_of_month(df, date_column_name):
    # convert to datetime format
    starting_month_list = df[date_column_name].apply(lambda x: date(x.year, x.month, 1))
    return starting_month_list


def get_week_of_month(df, date_column_name) -> list:

    def compute_week_of_month(date_value):
        first_day = date(date_value.year, date_value.month, 1)
        offset = (date_value.weekday() + 1 - first_day.weekday()) % 7
        week_of_month = (date_value.day + offset - 1) // 7 + 1
        return week_of_month

    week_of_month_list = df[date_column_name].apply(lambda x: compute_week_of_month(x))

    return week_of_month_list


def load_large_text_file(file_path):
    data_list = []
    with open(file_path, 'r') as file:
        for line in file:
            data_list.append(line.rstrip('\n'))
    return data_list



if __name__ == '__main__':
    
    # load, preprocess
    table_name = 'crypto'
    engine = create_engine(f'postgresql://postgres:postgres@localhost:5432/{table_name}')
    df_raw = pd.read_sql_query('select * from klines',con=engine)
    df_5min = df_raw.query("period_type == '5min'").reset_index()
    df_5min = get_day_of_week(df_5min)
    df_5min = set_action(df_5min)
    df_5min = pareto_distribution_bins(df_5min, get_optimal_pareto_distribution_num_bins(df_5min))
    df_5min = encode_time(df_5min)
    df = df_5min.copy()

    rewards = df['one_time_reward'].to_list()
    df_volumes = df[desired_col]
    df_volumes.columns = renamed_col

    df_volumes['starting_month'] = convert_to_first_day_of_month(df, 'date')
    df_volumes['week_of_month'] = get_week_of_month(df, 'date')

    # **Start aggregating the trading data**
    # ***Based on year, month, day, time to see the max, min, average, median, sd of the price and volumes***
    volume_cols = ['week_of_month', 'local_numeric_day', 'encoded_time', 'trade_volumes']
    price_cols = ['week_of_month', 'local_numeric_day', 'encoded_time', 'trade_volumes', 'trade_total_price']
    reward_cols = ['week_of_month', 'local_numeric_day', 'encoded_time', 'sell_rewards', 'buy_rewards', 'actions']

    groupby_keys = ['week_of_month', 'local_numeric_day', 'encoded_time']

    df_volume_stats = df_volumes[volume_cols].groupby(groupby_keys).describe().reset_index()

    df_price = df_volumes[price_cols].groupby(groupby_keys).sum()
    df_price['daily_average_trade_total_price'] = df_price['trade_total_price'] / df_price['trade_volumes']
    df_price = df_price.drop(columns=['trade_volumes', 'trade_total_price'])
    df_price_stats = df_price.groupby(groupby_keys).mean()

    # ***Create nested reward table***
    # ***Keep the df_price_stats and df_volume_stats for reference***
    # ***Start creating nested reward table from here***
    df_reward_stats = df_volumes[reward_cols[:-1]].groupby(groupby_keys).describe().reset_index()
    df_sell_rewards = df_reward_stats['sell_rewards'][['mean']].copy()
    df_buy_rewards = df_reward_stats['buy_rewards'][['mean']].copy()
    sell_cols = ['sell_rewards']
    buy_cols = ['buy_rewards']
    df_sell_rewards.columns = sell_cols
    df_buy_rewards.columns = buy_cols

    #df_nested_rewards = pd.merge([df_reward_stats[['week_of_month', 'local_numeric_day', 'encoded_time']], df_sell_rewards, df_buy_rewards],).reset_index()
    df_nested_rewards = pd.concat([df_reward_stats[['week_of_month', 'local_numeric_day', 'encoded_time']], df_sell_rewards, df_buy_rewards], axis=1)

    # rename column to remove the tuple-like hierachy syntax for easier retrieve
    rename_cols = ['week_of_month', 'local_numeric_day', 'encoded_time', 'sell_rewards', 'buy_rewards']
    df_nested_rewards.columns = rename_cols

    # Pivot the DataFrame
    df_pivoted_rewards = pd.pivot_table(df_nested_rewards, values=['sell_rewards', 'buy_rewards'], index=['week_of_month', 'local_numeric_day', 'encoded_time'],
                                aggfunc=fill_values).reset_index()
    df_pivoted_rewards = df_pivoted_rewards.rename(columns={'sell_rewards': 'sell_action', 'buy_rewards': 'buy_action'})


    # assign reverse action reward for NaN value 
    df_pivoted_rewards['buy_action'] = df_pivoted_rewards['buy_action'].fillna(df_pivoted_rewards['sell_action']*-1)
    df_pivoted_rewards['sell_action'] = df_pivoted_rewards['sell_action'].fillna(df_pivoted_rewards['buy_action']*-1)


    # add in no action reward value 
    df_pivoted_rewards['no_action'] = 0
    df_pivoted_rewards['label_encoded_time'] = label_encoder.fit_transform(df_pivoted_rewards[['encoded_time']])

    # get the unique value of each column for each state 
    state_unique_counts = df_pivoted_rewards.nunique()

    # initialize shape size
    state_array_shape = tuple(state_unique_counts[:3])
    # add 3 unique actions 
    state_array_shape += (num_action ,)
    print("State array shape:", state_array_shape)

    # create the array with the initialized shape size 
    state_array = np.zeros(state_array_shape)

    # start padding reward value into each state respectively
    # Iterate over the rows of the DataFrame
    for index, row in df_pivoted_rewards.iterrows():
        week_index = row['week_of_month']-1
        day_index = row['local_numeric_day']-1
        time_index = row['label_encoded_time']-1
        value = [row['buy_action'], row['sell_action'], row['no_action']]
        state_array[week_index, day_index, time_index] = value


    # assign state array to be reward array for easier reference
    reward_table = state_array.copy()


    # Start modifying the SARSA nested state iteration from here
    # Training loop
    environments_list = []
    total_rewards_list = []
    rewards_list = []
    steps_list = []
    num_episodes = 10000
    # num_steps_per_episode = 1000
    GAMMA = 0.9
    isVerbose = False

    for episode in range(num_episodes):

        print("\nEpisode:", episode)

        # initialize cumulative rewards
        total_rewards = 0
        steps = []
        environments = []
        rewards = []

        current_state = (0,0,0) # Starting state
        action, action_value = policy(Q, sarsa_agent, current_state, epsilon, verbose=isVerbose)
        # update upcoming allowed actions
        if action == 0:
            sarsa_agent.isHolding = True
        else:
            sarsa_agent.isHolding = False

        rewards_value = reward_table[current_state][action]
        total_rewards += rewards_value

        steps.append(action)
        rewards.append(rewards_value)
        environments.append(current_state)

        # when current state has not iterate until the last row of Q table
        # while (current_state != (reward_table.shape[0],reward_table.shape[1],reward_table.shape[2])):

        for month in range(reward_table.shape[0]):
            for day in range(reward_table.shape[1]):
                for time in range(reward_table.shape[2]):

                    # when iterating (0,0) start from (0,0,1) to (0,0,287) because (0,0,0) already initialized on top
                    if current_state[0] == 0 and current_state[1] == 0:
                        current_state = [month, day, time + 1]
                    else:
                        current_state = [month, day, time]

                    current_state = tuple(current_state)
                    # print("Current state:", current_state)
                    # print("Current action:", action, action_value)

                    rewards_value = reward_table[current_state][action]
                    total_rewards += rewards_value

                    # print("Total rewards:", total_rewards)

                    steps.append(action)
                    rewards.append(rewards_value)
                    environments.append(current_state)

                    # if not last row of state then + 1 else move up 1 level then + 1
                    if current_state[2] < reward_table.shape[2] - 1:
                        next_state = [current_state[0], current_state[1], current_state[2] + 1]
                        # print("In time:", current_state, next_state)
                    elif reward_table.shape[2] - 1 == current_state[2] and current_state[1] < reward_table.shape[1] - 1:
                        next_state = [current_state[0], current_state[1] + 1, current_state[2]]
                        # print("In day:", current_state,  next_state)
                    elif current_state[0] < reward_table.shape[0] - 1:
                        next_state = [current_state[0] + 1, current_state[1], current_state[2]]
                        # print("In month:", current_state, next_state)
                    
                    next_state = tuple(next_state)
                    next_action, next_action_value = policy(Q, sarsa_agent, next_state, epsilon, verbose=isVerbose)

                    #print("End:", current_state, action, rewards, next_state, next_action)
                    update_q_value(current_state, action, rewards, rewards_value, next_state, next_action, verbose=False)

                    # print(f'After update:, action: {action}, action_value : {action_value}, next_action: {next_action}, next_action_value: {next_action_value}\n')
                    # update upcoming allowed actions
                    if next_action == 0:
                        sarsa_agent.isHolding = True
                    if next_action == 1:
                        sarsa_agent.isHolding = False

                    current_state = next_state
                    action = next_action 
                    action_value = next_action_value

        total_rewards_list.append(sum(rewards))
        rewards_list.append(rewards)
        steps_list.append(steps)
        environments_list.append(environments)


    df_check = pd.DataFrame(total_rewards_list, columns=['rewards'])

    max_value_index = total_rewards_list.index(max(total_rewards_list))
    worst_value_index = total_rewards_list.index(min(total_rewards_list))

    print(max_value_index)
    print(worst_value_index)
    print(max(total_rewards_list))
    print(min(total_rewards_list))
    print("Max step list:", steps_list[max_value_index])
    print("Bad step list:", steps_list[worst_value_index])


    # see the main action difference between max and worst step at what timestep
    df_compare = pd.DataFrame(steps_list[max_value_index], columns=['max_steps_action'])
    df_compare['worst_steps_action'] = steps_list[worst_value_index]
    df_compare['is_same'] = df_compare['max_steps_action'].equals(df_compare['worst_steps_action'])


    joblib.dump(Q, 'sarsa_crypto.joblib')
    # [Q, total_rewards_list, rewards_list, steps_list]

    # save rewards and steps list
    with open('rewards.txt', 'w') as file:
        # Convert each element in the list to a string and write it to the file
        for item in rewards_list:
            file.write(str(item) + '\n')


    with open('steps.txt', 'w') as file:
        # Convert each element in the list to a string and write it to the file
        for item in steps_list:
            file.write(str(item) + '\n')


    Q = joblib.load('sarsa_crypto.joblib')
    rewards_list = load_large_text_file('rewards.txt')
    steps_list = load_large_text_file('steps.txt')


    # Load the trained SARSA model
    model = joblib.load('sarsa_crypto.joblib')


    # Load and preprocess your test data
    test_data = load_test_data()  # Replace with your code to load test data
    preprocessed_data = preprocess(test_data)  # Replace with your code to preprocess the data

    # Evaluate model performance on test data
    total_rewards = 0
    num_episodes = len(preprocessed_data)

    for episode in preprocessed_data:
        state = episode['state']
        done = False
        episode_reward = 0

        while not done:
            action = model.predict(state)
            next_state, reward, done = environment.step(action)
            episode_reward += reward
            state = next_state

        total_rewards += episode_reward

    average_reward = total_rewards / num_episodes
    print(f"Average reward on test data: {average_reward}")
























