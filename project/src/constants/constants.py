# the trade volumes here originally is "amount", total usd price is "vol"
# will rename after retrieved from the dataframe

########################################
#  preprocessing.py requires variable  #
########################################
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