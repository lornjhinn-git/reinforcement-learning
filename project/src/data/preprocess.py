import pandas as pd 
import numpy as np
import random
import math
from datetime import date 
from sklearn.preprocessing import LabelEncoder 

# features that need to be scaled

# 1. month (jan - dec)
# 2. day (mon - sat)
# 3. time (12 am - 12 pm every 5 min)
# 4. actions (buy, sell, no action)

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