# the trade volumes here originally is "amount", total usd price is "vol"
# will rename after retrieved from the dataframe
import os
import logging

#####################
# General reference #
#####################
ACTION_DICT = {
	'buy': 0,
	'sell': 1,
	'no_action': 2
}

INDEX_TO_ACTION = {
	0: "Buy",
	1: "Sell",
	2: "No Action"
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



def setup_logger(log_file):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # Create a FileHandler with the specified log file path
    file_handler = logging.FileHandler(log_file, mode='w')  # 'a' appends to the file
    file_handler.setLevel(logging.DEBUG)
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add the FileHandler to the logger
    logger.addHandler(file_handler)
    
    return logger

LOG_DIR = './log'
SARSA_LOG_FILE = os.path.join(LOG_DIR, "sarsa_log.log")
SARSA_LOGGER = setup_logger(SARSA_LOG_FILE)
# SARSA_LOG_LEVEL = logging.DEBUG
# SARSA_LOGGER = logging.basicConfig(
#     level=logging.DEBUG,
#     filename = SARSA_LOG_FILENAME,
#     filemode="w+",
#     format="%(asctime)-15s %(levelname)-8s %(message)s"
# )


# Note: is important to put in sequential order
STATE_COLUMNS = [
    'week_of_month',
    'local_numeric_day',
    'label_encoded_time'
]

# total unique value for each state
STATE_DICT = {
    'week_of_month': 5,
    'local_numeric_day': 7,
    'label_encoded_time': 288
}