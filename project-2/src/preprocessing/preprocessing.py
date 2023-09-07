# Import general package 
from ..sarsa import *
from ..config import constants as Constantor
from ..config.logger import verbose_logger

from datetime import date, datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder 


import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")


# features that need to be scaled
# print("OK")

# 1. month (jan - dec)
# 2. day (mon - sat)
# 3. time (12 am - 12 pm every 5 min)
# 4. actions (buy, sell, no action)

encoder = LabelEncoder()

def get_day_of_week(df) -> pd.DataFrame:
    """
    Returns the name of the day of the week for the given day number (0-6)
    """
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df['local_numeric_day'] =  df['datetime'].apply(lambda x: (x.weekday()) % 7)
    df['local_day'] =  df['local_numeric_day'].apply(lambda x: days[x])
    return df 


def convert_str_to_datetime(input_value):
    if isinstance(input_value, str):
        try:
            datetime_value = datetime.strptime(input_value, '%Y-%m-%d %H:%M:%S')  # Change format if needed
            return datetime_value
        except ValueError:
            pass
    return input_value


def set_action(df, optimum_sell_rewards=15, optimum_buy_rewards=15) -> pd.DataFrame:
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
def get_optimal_normal_distribution_num_bins(df) -> pd.DataFrame:
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
def get_optimal_pareto_distribution_num_bins(df) -> pd.DataFrame:
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


def pareto_distribution_bins(df) -> pd.DataFrame:
    """Creates power law bins for the 'volume_trade' column of the given
    DataFrame using the qcut function, and returns the updated DataFrame.
    """

    # Compute the optimal number of bins for quantiles splitting
    # num_bins = get_optimal_pareto_distribution_num_bins(df)
    num_bins = Constantor.PARETO_NUM_BINS

    # Compute the quantiles of the 'volume_trade' column using a power law distribution
    quantiles = pd.qcut(df['amount'], num_bins, labels=False, duplicates='drop')

    # Add a new column to the DataFrame with the bin labels
    df['volume_bins'] = quantiles

    # Return the updated DataFrame
    return df


def encode_time(df) -> pd.DataFrame:
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
    df['label_encoded_time'] = encoder.fit_transform(df[['encoded_time']])


    # Drop the original hour, minute, and second columns
    df = df.drop(['hour', 'minute', 'second', 'hour_str', 'minute_str', 'second_str'], axis=1)

    # Return the updated DataFrame with the encoded time string
    return df


def convert_to_first_day_of_month(df) -> pd.DataFrame:
    # convert to datetime format
    date_column_name = 'date'
    assign_column_name = 'starting_month'
    starting_month_list = df[date_column_name].apply(lambda x: date(x.year, x.month, 1))
    df.loc[:, assign_column_name] = starting_month_list

    return df


def get_week_of_month(df) -> pd.DataFrame:

    def compute_week_of_month(date_value):
        first_day = date(date_value.year, date_value.month, 1)
        offset = (date_value.weekday() + 1 - first_day.weekday()) % 7
        week_of_month = ((date_value.day + offset - 1) // 7 + 1) -1
        return week_of_month

    date_column_name = 'date'
    assign_column_name = 'week_of_month'
    week_of_month_list = df[date_column_name].apply(lambda x: compute_week_of_month(x))
    df.loc[:, assign_column_name] = week_of_month_list

    return df


def get_daily_average_trade_total_price(df) -> pd.DataFrame:
    df = df[Constantor.PRICE_COLS].groupby(Constantor.GROUPBY_KEYS).sum()
    df['daily_average_trade_total_price'] = df['trade_total_price'] / df['trade_volumes']
    df = df.drop(columns=['trade_volumes', 'trade_total_price'])
    return df


def get_period_mean_price(df) -> pd.DataFrame:
    df['average_period_price'] = round(df['vol'] / df['amount'], 2)
    return df 


# Create a custom aggregation function to fill in values based on conditions
def fill_values(column):
    if column[column > 0].empty:
        return None
    return column[column > 0].values[0]


def create_price_table(df, Q):
    # get the unique value of each column for each state 
    price_array = np.zeros((5,7,288,1))
    df['label_encoded_time'] = encoder.fit_transform(df[['encoded_time']])
    df = df[['week_of_month', 'local_numeric_day', 'label_encoded_time', 'average_period_price']]\
        .groupby(['week_of_month', 'local_numeric_day', 'label_encoded_time'])\
        .mean().reset_index()

    # Iterate over the rows of the DataFrame
    for index, row in df.iterrows():
        week_index = int(row['week_of_month'])
        day_index = int(row['local_numeric_day'])
        time_index = int(row['label_encoded_time'])
        value = row['average_period_price']
        if value == 0: print(week_index, day_index, time_index) 
        price_array[week_index, day_index, time_index] = value

    return price_array


def create_reward_table(df_reward_stats, verbose=False) -> tuple[np.array, np.array]:
    df_sell_rewards = df_reward_stats['sell_rewards'][['mean']].copy()
    df_buy_rewards = df_reward_stats['buy_rewards'][['mean']].copy()
    df_sell_rewards.columns = ['sell_rewards']
    df_buy_rewards.columns = ['buy_rewards']
    df_nested_rewards = pd.concat([df_reward_stats[['week_of_month', 'local_numeric_day', 'encoded_time']], df_sell_rewards, df_buy_rewards], axis=1)

    # rename column to remove the tuple-like hierachy syntax for easier retrieve
    rename_cols = ['week_of_month', 'local_numeric_day', 'encoded_time', 'sell_rewards', 'buy_rewards']
    df_nested_rewards.columns = rename_cols

    df_pivoted_rewards = pd.pivot_table(df_nested_rewards, values=['sell_rewards', 'buy_rewards'], index=['week_of_month', 'local_numeric_day', 'encoded_time'],
                                aggfunc=fill_values).reset_index()
    df_pivoted_rewards = df_pivoted_rewards.rename(columns={'sell_rewards': 'sell_action', 'buy_rewards': 'buy_action'})

    # assign reverse action reward for NaN value 
    df_pivoted_rewards['buy_action'] = df_pivoted_rewards['buy_action'].fillna(df_pivoted_rewards['sell_action']*-1)
    df_pivoted_rewards['sell_action'] = df_pivoted_rewards['sell_action'].fillna(df_pivoted_rewards['buy_action']*-1)
    df_pivoted_rewards['no_action'] = 0
    df_pivoted_rewards['label_encoded_time'] = encoder.fit_transform(df_pivoted_rewards[['encoded_time']])

    # get the unique value of each column for each state 
    state_unique_counts = df_pivoted_rewards.nunique()

    # initialize shape size
    state_array_shape = tuple(state_unique_counts[:Constantor.NUM_ACTION])
    # add 3 unique actions 
    state_array_shape += (Constantor.NUM_ACTION ,)
    # print("State array shape:", state_array_shape)

    # create the array with the initialized shape size 
    state_array = np.zeros(state_array_shape)

    # start padding reward value into each state respectively
    # Iterate over the rows of the DataFrame
    for index, row in df_pivoted_rewards.iterrows():
        week_index = int(row['week_of_month'])
        day_index = int(row['local_numeric_day'])
        time_index = int(row['label_encoded_time'])
        value = [row['buy_action'], row['sell_action'], row['no_action']]
        state_array[week_index, day_index, time_index] = value

    # 25/7/2023: Temporarily add in the two dimensional holding or not holding into the state.
    # In the future, all this categorical or continuous factor should be written in a scalable way instead of so ad hoc
    holding_array = np.array([[0,1]]).T
    reward_table  = holding_array[:, np.newaxis, np.newaxis, np.newaxis, :] + state_array

    # assign state array to be reward array for easier reference
    # 11/8/2023: Temporarily ignore old reward table as Q dimension. Directly update state array as Q dimension
    Q = np.zeros(state_array.shape)

    return reward_table, Q

@verbose_logger
def preprocessing(df) -> pd.DataFrame:

	# convert str format datetime value to datetime format
	df['datetime'] = df['datetime'].apply(lambda x: convert_str_to_datetime(x))

	df_preprocessed = df.pipe(get_day_of_week)\
						.pipe(set_action)\
						.pipe(pareto_distribution_bins)\
						.pipe(encode_time)\
                        .pipe(get_period_mean_price)\
                        .pipe(convert_to_first_day_of_month)\
	 					.pipe(get_week_of_month)\
                        .pipe(convert_to_first_day_of_month)\
	 					.pipe(get_week_of_month)
	
	df_preprocessed_volumes = df_preprocessed[Constantor.DESIRED_COL]
	df_preprocessed_volumes.columns = Constantor.RENAMED_COL
	df_preprocessed_volumes = df_preprocessed_volumes.pipe(convert_to_first_day_of_month)\
	 												 .pipe(get_week_of_month)

	df_preprocessed_prices = get_daily_average_trade_total_price(df_preprocessed_volumes)

	df_volume_stats = df_preprocessed_volumes[Constantor.VOLUME_COLS].groupby(Constantor.GROUPBY_KEYS).describe().reset_index()
	df_price_stats = df_preprocessed_volumes[Constantor.PRICE_COLS].groupby(Constantor.GROUPBY_KEYS).mean()
	df_reward_stats = df_preprocessed_volumes[Constantor.REWARD_COLS[:-1]].groupby(Constantor.GROUPBY_KEYS).describe().reset_index()

    # 10/8/2023: Temporarily comment out as I am implementing a new way of reward function instead of restricting
	# # generate reward table & initialize Q table
	# reward_table, Q = create_reward_table(df_reward_stats)
	
	# Sarsa.reward_table = reward_table
	# Sarsa.Q = Q

	return df_preprocessed, df_reward_stats, df_price_stats, df_volume_stats, df_preprocessed_prices
