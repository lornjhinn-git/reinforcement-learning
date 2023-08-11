from .. import sarsa
from ..constants import constants as Constantor
from ..preprocessing import preprocessing as Preprocessor, utils as Utilator
import pandas as pd 
import numpy as np
from sqlalchemy import create_engine
import math
import argparse
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parameters for SARSA")
    
    # Adding command line arguments
    parser.add_argument("--num_train_episodes", type=int, help="Num of episodes to train")
    parser.add_argument("--num_test_episodes", type=int, help="Num of episodes to validate")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--gamma", type=float, help="Determine for quick or long term reward. Higher the longer")
    parser.add_argument("--epsilon", type=float, help="Probability to explore")
    parser.add_argument("--budget", type=int, help="Game budget to begin")
    # parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")
    
    # Parse the arguments
    return parser.parse_args()


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


def train():
	engine = create_engine(f'postgresql://postgres:postgres@localhost:5432/{Constantor.DATABASE_NAME}')

	df = pd.read_sql_query(f'select * from {Constantor.TABLE_NAME}',con=engine)\
			   .query(f"period_type == '{Constantor.PERIOD_TYPE}'").reset_index()
	
	# Split data for train test
	train_test_split(df)
	df_preprocessed_train, df_reward_stats = Preprocessor.preprocessing(Sarsa.train_data)
	_, Sarsa.Q = Preprocessor.create_reward_table(df_reward_stats)
	Sarsa.price_table = Preprocessor.create_price_table(df_preprocessed_train, Sarsa.Q)

	print("Sarsa.Q:", Sarsa.Q.shape)

	Sarsa.train(df_preprocessed_train)
	Sarsa.save_model()
	#Sarsa.validate()

	# df_preprocessed_train.to_csv("./validation/preprocessed_data.csv")
	# df_reward_stats.to_csv("./validation/reward_stats_data.csv")


if __name__ == '__main__':
	print("Start training process")
	global Sarsa 
	if len(sys.argv) == 1:
		Sarsa = sarsa.SARSA()
		train()
	else: 
		args = parse_arguments()
		Sarsa = sarsa.SARSA(**vars(args))
		train()
