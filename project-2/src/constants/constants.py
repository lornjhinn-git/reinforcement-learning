# the trade volumes here originally is "amount", total usd price is "vol"
# will rename after retrieved from the dataframe


#####################
# General reference #
#####################
ACTION_DICT = {
	'buy': 0,
	'sell': 1,
	'no_action': 2
}


###############################
# train.py requires variable  #
###############################
DATABASE_NAME = 'crypto'
TABLE_NAME = 'klines'
PERIOD_TYPE = '5min'
PARETO_NUM_BINS = 10

TRAIN_SPLIT = 0.8
NUM_ACTION = 3 # buy, sell, no action


########################################
#  preprocessing.py requires variable  #
########################################
DESIRED_COL = [ 
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


RENAMED_COL = [ 
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


VOLUME_COLS = [
    'week_of_month', 
    'local_numeric_day', 
    'encoded_time', 
    'trade_volumes'
]


PRICE_COLS = [
    'week_of_month', 
    'local_numeric_day', 
    'encoded_time', 
    'trade_volumes', 
    'trade_total_price'
]


REWARD_COLS = [
    'week_of_month', 
    'local_numeric_day', 
    'encoded_time', 
    'sell_rewards', 
    'buy_rewards', 
    'actions'
]


GROUPBY_KEYS = [
    'week_of_month', 
    'local_numeric_day', 
    'encoded_time'
]