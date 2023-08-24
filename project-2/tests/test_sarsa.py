import src.sarsa as Sarsa
import src.preprocessing.preprocessing as Preprocessor
import src.config.constants as Constantor

import pandas as pd 
import random
import joblib
import numpy as np

sarsa_tester = Sarsa.SARSA()

def test_update_top_n_values():
    test_episode = 1 #sarsa_tester.keep_top_n_steps
    test_values = 10081 # the total encoded time
    for episode in range(test_episode):
        print("Episode:", episode)
        random_steps_values = [random.choice([2, 0, 1]) for _ in range(test_values)]
        random_rewards_values = [random.choice([-1, 0, 1]) for _ in range(test_values)]
        # print("random_steps_values", len(random_steps_values), random_steps_values)
        # print("random reward values", len(random_rewards_values), random_rewards_values, "\n")
        sarsa_tester.update_top_n_values(random_rewards_values, random_steps_values)
        print("df_steps:", sarsa_tester.df_steps)
        print("df_rewards:", sarsa_tester.df_rewards)
        print("df_steps columns:", sarsa_tester.df_steps.columns)
        print("df_rewards columns:", sarsa_tester.df_rewards.columns, "\n")


def test_update_worst_n_values():
    test_episode = 1 #sarsa_tester.keep_top_n_steps
    test_values = 10081 # the total encoded time
    for episode in range(test_episode):
        print("Episode:", episode)
        random_steps_values = [random.choice([2, 0, 1]) for _ in range(test_values)]
        random_rewards_values = [random.choice([-1, 0, 1]) for _ in range(test_values)]
        # print("random_steps_values", len(random_steps_values), random_steps_values)
        # print("random reward values", len(random_rewards_values), random_rewards_values, "\n")
        sarsa_tester.update_worst_n_values(random_rewards_values, random_steps_values)
        print("df_steps:", sarsa_tester.df_steps)
        print("df_rewards:", sarsa_tester.df_rewards)
        print("df_steps columns:", sarsa_tester.df_steps.columns)
        print("df_rewards columns:", sarsa_tester.df_rewards.columns, "\n")


def test_validate():

    from collections import Counter

    value_estimates = []

    sarsa_tester.Q = joblib.load('./validation/model/sarsa_crypto_backup.joblib')
    value_table = np.max(sarsa_tester.Q, axis=-1)
    value_step = np.argmax(sarsa_tester.Q, axis=-1).tolist()
    print("value table:", value_table.shape)
    print("value step:", len(value_step))

    def flatten_array(arr):
        flat_list = []
        for item in arr:
            if isinstance(item, list):
                flat_list.extend(flatten_array(item))
            else:
                flat_list.append(item)
        return flat_list

    flatten_value_steps = flatten_array(value_step)

    sarsa_tester.test_data = pd.read_csv("./validation/data/test_data.csv")
    df_preprocessed_test, _ = Preprocessor.preprocessing(sarsa_tester.test_data)
    df_preprocessed_test = df_preprocessed_test[Constantor.STATE_COLUMNS].drop_duplicates()

    def combine_columns(row):
        return (row[Constantor.STATE_COLUMNS[0]], row[Constantor.STATE_COLUMNS[1]], row[Constantor.STATE_COLUMNS[2]])
    df_preprocessed_test['states'] = df_preprocessed_test.apply(combine_columns, axis=1)
    for _, state in enumerate(df_preprocessed_test['states']):
        update_state = tuple(value - 1 for value in state)
        value_estimate = value_table[update_state]
        value_estimates.append(value_estimate)

    average_value = np.mean(value_estimates)
    print("Average value estimate:", average_value)

    step_counter = Counter(flatten_value_steps)
    for value, count in step_counter.items():
        print(f"{value}: {count} times")