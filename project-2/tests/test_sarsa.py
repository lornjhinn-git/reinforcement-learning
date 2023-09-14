import src.agent as Agent
import src.preprocessing.preprocessing as Preprocessor
import src.config.constants as Constantor

import pandas as pd 
import random
import joblib
import numpy as np

sarsa = Agent.Sarsa()

# def test_update_top_n_values():
#     test_episode = 1 #sarsa.keep_top_n_steps
#     test_values = 10081 # the total encoded time
#     for episode in range(test_episode):
#         print("Episode:", episode)
#         random_steps_values = [random.choice([2, 0, 1]) for _ in range(test_values)]
#         random_rewards_values = [random.choice([-1, 0, 1]) for _ in range(test_values)]
#         # print("random_steps_values", len(random_steps_values), random_steps_values)
#         # print("random reward values", len(random_rewards_values), random_rewards_values, "\n")
#         sarsa.update_top_n_values(random_rewards_values, random_steps_values)
#         print("df_steps:", sarsa.df_steps)
#         print("df_rewards:", sarsa.df_rewards)
#         print("df_steps columns:", sarsa.df_steps.columns)
#         print("df_rewards columns:", sarsa.df_rewards.columns, "\n")


# def test_update_worst_n_values():
#     test_episode = 1 #sarsa.keep_top_n_steps
#     test_values = 10081 # the total encoded time
#     for episode in range(test_episode):
#         print("Episode:", episode)
#         random_steps_values = [random.choice([2, 0, 1]) for _ in range(test_values)]
#         random_rewards_values = [random.choice([-1, 0, 1]) for _ in range(test_values)]
#         # print("random_steps_values", len(random_steps_values), random_steps_values)
#         # print("random reward values", len(random_rewards_values), random_rewards_values, "\n")
#         sarsa.update_worst_n_values(random_rewards_values, random_steps_values)
#         print("df_steps:", sarsa.df_steps)
#         print("df_rewards:", sarsa.df_rewards)
#         print("df_steps columns:", sarsa.df_steps.columns)
#         print("df_rewards columns:", sarsa.df_rewards.columns, "\n")


# def test_validate():

#     from collections import Counter

#     value_estimates = []

#     sarsa.Q = joblib.load('./validation/model/sarsa_crypto_backup.joblib')
#     value_table = np.max(sarsa.Q, axis=-1)
#     value_step = np.argmax(sarsa.Q, axis=-1).tolist()
#     print("value table:", value_table.shape)
#     print("value step:", len(value_step))

#     def flatten_array(arr):
#         flat_list = []
#         for item in arr:
#             if isinstance(item, list):
#                 flat_list.extend(flatten_array(item))
#             else:
#                 flat_list.append(item)
#         return flat_list

#     flatten_value_steps = flatten_array(value_step)

#     sarsa.test_data = pd.read_csv("./validation/data/test_data.csv")
#     df_preprocessed, _, _, _, _ = Preprocessor.preprocessing(sarsa.test_data)
#     df_preprocessed = df_preprocessed[Constantor.STATE_COLUMNS].drop_duplicates()

#     def combine_columns(row):
#         return (row[Constantor.STATE_COLUMNS[0]], row[Constantor.STATE_COLUMNS[1]], row[Constantor.STATE_COLUMNS[2]])
#     df_preprocessed['states'] = df_preprocessed.apply(combine_columns, axis=1)
#     for _, state in enumerate(df_preprocessed['states']):
#         update_state = tuple(value - 1 for value in state)
#         value_estimate = value_table[update_state]
#         value_estimates.append(value_estimate)

#     average_value = np.mean(value_estimates)
#     print("Average value estimate:", average_value)

#     step_counter = Counter(flatten_value_steps)
#     for value, count in step_counter.items():
#         print(f"{value}: {count} times")


def test_validate():

    from collections import Counter

    sarsa.test_data = pd.read_csv("./validation/data/test_data.csv")
    df_preprocessed, _, _, _, _ = Preprocessor.preprocessing(sarsa.test_data)
    sarsa.price_table = Preprocessor.create_price_table(df_preprocessed, sarsa.Q)
    sarsa.Q = joblib.load('./validation/model/sarsa_crypto_backup.joblib')

    # reserved only the data that has complete info on price table
    # the first column will always be the key for retrieve
    # 7/9/2023 2 scenarios: 
    # 1. When predicting future without aggregating past price table, will only able to assess the days with complete price table info
    # 2. If want to able to predict full future data, then need to use newly updated aggregate price table
    # Thus, in future will need a indicator to decide whether to assess partial or full, if full then average full price table 
    # without dedicately filter out
    assess_months_list = (
        df_preprocessed[Constantor.STATE_COLUMNS].drop_duplicates().groupby(Constantor.STATE_COLUMNS[0])\
        .nunique().reset_index().query(f"{Constantor.STATE_COLUMNS[1]} == {Constantor.STATE_DICT.get(Constantor.STATE_COLUMNS[1])} and {Constantor.STATE_COLUMNS[2]} == {Constantor.STATE_DICT.get(Constantor.STATE_COLUMNS[2])}")\
        [Constantor.STATE_COLUMNS[0]].tolist()
    )
    df_state = df_preprocessed[Constantor.STATE_COLUMNS].loc[df_preprocessed[Constantor.STATE_COLUMNS[0]].isin(assess_months_list)]
    print("df_state preview:", df_state.head())

    # transform into state list for evaluation
    df_state['state_list'] = df_state.apply(lambda row: row.tolist(), axis=1)
    state_list = df_state['state_list']

    # step_list = np.argmax(sarsa.Q, axis=-1).tolist()
    # price_list = np.max(sarsa.price_table, axis=-1).tolist()
    # step_list = np.argmax(sarsa.Q, axis=-1)
    # price_list = np.max(sarsa.price_table, axis=-1)

    # def flatten_array(arr):
    #     flat_list = []
    #     for item in arr:
    #         if isinstance(item, list):
    #             flat_list.extend(flatten_array(item))
    #         else:
    #             flat_list.append(item)
    #     return flat_list
    # flatten_price = flatten_array(price_list)
    # flatten_step = flatten_array(step_list) 

    # def flatten_array_test(arr):
    #     flatten_arr = []
    #     for week in range(arr.shape[0]):
    #         for day in range(arr.shape[1]):
    #             for time in range(arr.shape[2]):
    #                 #print(week, day, time)
    #                 flatten_arr.append(arr[week][day][time])

    #     return flatten_arr
    # flatten_price = flatten_array_test(price_list)
    # flatten_step = flatten_array_test(step_list) 

    print("Sarsa price table dimension:", sarsa.price_table.shape)
    print("State list preview:", state_list[:5])
    # print(len(flatten_price))
    # print(len(flatten_step))

    total_profit = 0
    single_profit = 0
    total_profit_list = []
    unit = 0
    budget = sarsa.budget
    buy = 0
    sell = 0
    no_action = 0

    for state in state_list:
        state = tuple(state)
        action = np.argmax(sarsa.Q[state])
        print("State:", state, "Action:", action)
        if action == 0 and budget > sarsa.price_table[state]: # buy
            budget -= sarsa.price_table[state]
            total_profit -= sarsa.price_table[state]
            total_profit_list.append(total_profit)
            unit += (1*0.998)
            buy += 1
        elif action == 1 and unit > 0: # sell
            budget += unit * sarsa.price_table[state] * 0.998
            total_profit += unit * sarsa.price_table[state] * 0.998
            total_profit_list.append(total_profit)
            print("Sell price:", sarsa.price_table[state], "total_profit:", total_profit)
            unit = 0
            sell += 1

        elif action == 2: # no action
            no_action += 1
            pass
        else:
            if action == 0:
                print(f"Couldnt perform action {Constantor.INDEX_TO_ACTION.get(action)} due to budget = {budget}")
            else:
                print(f"Couldnt perform action {Constantor.INDEX_TO_ACTION.get(action)} due to unit = {unit}")
            pass

        print("Action:", action, "Budget:", budget, "Price:", sarsa.price_table[state], "Total total_profit:", total_profit)

    potential_max_total_profit = [x for x in total_profit_list if x > 0]
    print("Len potential max total_profit:", potential_max_total_profit)
    if len(potential_max_total_profit) > 0:
        print("Average total_profit:", sum(potential_max_total_profit)/len(potential_max_total_profit))
    else:
        print("No total_profit")
    print(unit, budget, total_profit)
    print("Buy counter:", buy, "Sell counter:", sell, "No action counter:", no_action)


def test_filter():
    import pandas as pd

    # Sample list of dictionaries
    data = [
        {'name': 'John', 'age': 30},
        {'name': 'Alice', 'age': 25},
        pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}),  # DataFrame
        {'name': 'Bob', 'age': 28},
        pd.DataFrame({'X': ['a', 'b', 'c'], 'Y': ['d', 'e', 'f']}),  # DataFrame
    ]

    # Use a lambda function to filter out DataFrames and objects
    filtered_data = list(filter(lambda x: isinstance(x, pd.DataFrame) or not isinstance(x, dict), data))

    print(filtered_data)