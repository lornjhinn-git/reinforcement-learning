from .. import agent as RL_Agent
from ..config import constants as Constantor
from ..config.logger import verbose_logger
from ..preprocessing import preprocessing as Preprocessor, utils as Utilator
import pandas as pd 
import numpy as np
from sqlalchemy import create_engine
import math
import argparse
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parameters for Agent")
    
    # Adding command line arguments
    parser.add_argument("--num_train_episodes", type=int, help="Num of episodes to train")
    parser.add_argument("--num_test_episodes", type=int, help="Num of episodes to validate")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--gamma", type=float, help="Determine for quick or long term reward. Higher the longer")
    parser.add_argument("--epsilon", type=float, help="Probability to explore")
    parser.add_argument("--budget", type=int, help="Game budget to begin")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")
    parser.add_argument("--debug", action="store_true", help="Enable to log output variables")
    parser.add_argument("--rl_algorithm", default='Sarsa', type=str, help="Specifiy algorithm type for training and evaluation, options: Q_Learning, Sarsa")

    # Parse the arguments
    return parser.parse_args()


@verbose_logger
# Generate the sample data
def train_test_split(df):
    global Agent
    Agent.train_data = df[:math.floor(df.shape[0]*Constantor.TRAIN_SPLIT)]
    Agent.test_data = df[math.floor(df.shape[0]*Constantor.TRAIN_SPLIT):]
    ##############
	# train data #
	##############
    Agent.train_data.to_csv("./validation/data/train_data.csv")
    Agent.train_data.to_csv(f"./validation/data/train_data_{Utilator.get_formatted_date()}.csv")
    ##############
	# test data  #
	##############
    Agent.test_data.to_csv("./validation/data/test_data.csv")
    Agent.test_data.to_csv(f"./validation/data/test_data_{Utilator.get_formatted_date()}.csv")
    # print(f"Total sample size: {df.shape}")
    # print(f"Agent train data sample size: {Agent.train_data.shape}")
    # print(f"Agent test data sample size: {Agent.test_data.shape}")


@verbose_logger
def train():
	engine = create_engine(f'postgresql://postgres:postgres@localhost:5432/{Constantor.DATABASE_NAME}')

	df = pd.read_sql_query(f'select * from {Constantor.TABLE_NAME}',con=engine)\
			   .query(f"period_type == '{Constantor.PERIOD_TYPE}'").reset_index()
	
	# Split data for train test
	train_test_split(df)
	df_preprocessed_train, df_reward_stats, _, _, _ = Preprocessor.preprocessing(Agent.train_data)
	_, Agent.Q = Preprocessor.create_reward_table(df_reward_stats)
	Agent.price_table = Preprocessor.create_price_table(df_preprocessed_train, Agent.Q)

	# print("Agent.Q:", Agent.Q.shape)

	Agent.train(df_preprocessed_train)
	Agent.save_model()
	# Agent.save_best_n_result()
	Agent.validate()

	# df_preprocessed_train.to_csv("./validation/preprocessed_data.csv")
	# df_reward_stats.to_csv("./validation/reward_stats_data.csv")


if __name__ == '__main__':
	if len(sys.argv) == 1:
		Agent = RL_Agent.Sarsa()
		train()
	else: 
		args = parse_arguments()
		filtered_args = vars(args).copy()
		if args.rl_algorithm == 'Sarsa':
			Agent = RL_Agent.Sarsa(**filtered_args)
		elif args.rl_algorithm == 'Q_Learning':
			Agent = RL_Agent.Q_Learning(**filtered_args)
		train()
