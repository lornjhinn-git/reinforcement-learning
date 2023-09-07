from .config import constants as Constantor
from .config.logger import class_verbose_logger
from .preprocessing import utils as Utilator
from .preprocessing import preprocessing as Preprocessor

from typing import Optional
import pandas as pd
import numpy as np
import random 
import joblib
import logging
from collections import Counter


logger = Constantor.SARSA_LOGGER


class SARSA:
    def __init__(
            self, 
            learning_rate=0.005,
            epsilon=0.1,
            gamma=0.9,
            num_train_episodes=5,
            num_test_episodes=1,
            data=None,
            Q=None,
            reward_table=None,
            budget=5000,
            price_table=None,
            purchase_unit=0,
            total_purchase_prices=0,
            purchase_states=[],
            train_data=None,
            test_data=None,
            train_value_dict = {
                            'environments': None,
                            'total_rewards': None,
                            'rewards': None, 
                            'steps': None
            },
            test_value_dict = {
                'environments': None,
                'total_rewards': None,
                'rewards': None, 
                'steps': None
	        },
            keep_top_n_steps=None,
            verbose=None
    ):
        self.learning_rate=0.005 if learning_rate is None else learning_rate
        self.epsilon=0.1 if epsilon is None else epsilon
        self.gamma=0.9 if gamma is None else gamma
        self.num_train_episodes=100 if num_train_episodes is None else num_train_episodes
        self.num_test_episodes=1 if num_test_episodes is None else num_test_episodes
        self.data=data
        self.Q=Q
        self.reward_table=reward_table
        self.budget=5000 if budget is None else budget  # set at 2k usd as starting budget
        self.price_table=price_table
        self.purchase_unit=0 if purchase_unit is None else purchase_unit
        self.total_purchase_prices=0 if total_purchase_prices is None else total_purchase_prices
        self.purchase_states=purchase_states
        self.train_data=train_data
        self.test_data=test_data
        self.train_value_dict=train_value_dict
        self.test_value_dict=test_value_dict
        self.keep_top_n_steps=100 if keep_top_n_steps is None else keep_top_n_steps
        self.last_state_dimension=288
        self.total_state_dimension=(288*5*7)+1
        self.df_steps=pd.DataFrame({f'Step_Top_{i+1}': [0]*self.total_state_dimension for i in range(self.keep_top_n_steps)})
        self.df_rewards=pd.DataFrame({f'Rewards_Top_{i+1}': [0]*self.total_state_dimension for i in range(self.keep_top_n_steps)})
        self.df_worst_steps = pd.DataFrame({f'Step_Worst_{i+1}': [0]*self.total_state_dimension for i in range(self.keep_top_n_steps)})
        self.df_worst_rewards = pd.DataFrame({f'Rewards_Worst_{i+1}': [0]*self.total_state_dimension for i in range(self.keep_top_n_steps)})
        self.top_n_total_rewards = [0]*self.keep_top_n_steps
        self.worst_n_total_rewards = [0]*self.keep_top_n_steps
        self.verbose = False if verbose is None else verbose


class SARSA(SARSA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("Child sarsa initialized")

        if self.verbose:
            logger.debug("Class initialized attributes:", self.__dict__)


    @class_verbose_logger
    def policy(self, state) -> tuple[int, float]: 
        r_var = random.random()
        explore = False
        explore = r_var < self.epsilon
        best_action = float('-inf')

        # filter allowed actions based on budget
        if self.purchase_unit > 0:
            allowed_actions = [1,2] # sell or no action
        elif self.purchase_unit == 0:
            allowed_actions = [0,2] # buy or no action
        else:
            allowed_actions = [0,1,2]

        if explore:
            # best_action = np.where(self.Q[state] == random.choice(self.Q[state]))[0][0]
            best_action = random.choice(allowed_actions)
            best_value = self.Q[state][best_action]
        else:
            for action in allowed_actions:
                if self.Q[state][action] > best_action:
                    best_action = action
                    best_value = self.Q[state][best_action]

        if self.verbose:
            logger.debug("Policying...")
            logger.debug(f"Current budget: {self.budget}, Holding Unit: {self.purchase_unit}, Total Purchase Price: {self.total_purchase_prices}")
            logger.debug(f"Random: {explore}, State: {state}, Action: {Constantor.INDEX_TO_ACTION.get(best_action)}, Action Value: {best_value}\n")
            print("\n")

        return best_action, best_value


    @class_verbose_logger
    def reward_function(self, state, action) -> float:
        if (
            # selling profit > average purchase price + trading fees
            action == 1 and 
            self.purchase_unit > 0 and 
            self.price_table[state] > ((self.total_purchase_prices/self.purchase_unit) + (self.price_table[state]*0.002))
        ): # making profit
            rewards_value = 1
        elif (
            (
                # selling profit <= average purchase price + trading fees
                action == 1 and 
                self.purchase_unit > 0 and 
                self.price_table[state] <= ((self.total_purchase_prices/self.purchase_unit) + (self.price_table[state]*0.002))
            ) or 
            (
                # total purchase more than budget 
                action == 0 and 
                self.purchase_unit > 0 and 
                self.total_purchase_prices > self.budget
            )
        ): # losing money
            rewards_value = -1.1
        else: # no action
            rewards_value = 0

        if action == 0: # buy 
            self.purchase_unit += 1
            self.total_purchase_prices += self.price_table[state]
            self.budget -= self.price_table[state]
        elif action == 1 and self.purchase_unit > 0: # sell
            self.budget += self.price_table[state]*0.998
            self.total_purchase_prices = 0
            self.purchase_unit = 0

        return rewards_value


    @class_verbose_logger
    # Update Q-value for a state-action pair based on observed rewards and estimated future Q-values
    def update_q_value(self, state:tuple, action:int, rewards_value:float, next_state:tuple, next_action:int):

        # Compute the updated Q-value using the self update equation
        current_q = self.Q[state][action]
        next_q = self.Q[next_state][next_action]
        new_q = current_q + self.learning_rate * (rewards_value + self.gamma * next_q - current_q)
        
        # Update the Q-value in the Q-table
        self.Q[state][action] = new_q

        if self.verbose:
            logger.debug("Updating...")
            logger.debug(f"Current Q: {current_q}, Rewards: {rewards_value}, Next Q: {next_q}, Updated current Q: {new_q}")
            logger.debug(f"Current state overall Q: {self.Q[state]}")
            print("\n")
    

    @class_verbose_logger
    # Update and keep the top n values 
    def update_top_n_values(self, rewards:list[float], steps:list[int]):
        for index, _ in enumerate(self.top_n_total_rewards):            
            if sum(rewards) > self.top_n_total_rewards[-1]:
                self.top_n_total_rewards.pop()
                self.top_n_total_rewards.append(sum(rewards))
                self.df_rewards.iloc[:, -1] = rewards
                self.df_steps.iloc[:, -1] = steps

            sorted_indices = [
                index for index, _ 
                in sorted(enumerate(self.top_n_total_rewards), key=lambda x: x[1], reverse=True)
            ]

            # re-sort the column based on sorted_indices
            current_rewards_columns = self.df_rewards.columns
            current_steps_columns = self.df_steps.columns
            sorted_rewards_columns = [
                current_rewards_columns[index] for index in sorted_indices
            ]
            sorted_steps_columns = [
                current_steps_columns[index] for index in sorted_indices
            ]

            self.df_rewards = self.df_rewards[sorted_rewards_columns]
            self.df_steps = self.df_steps[sorted_steps_columns]
            self.top_n_total_rewards = [self.top_n_total_rewards[index] for index in sorted_indices]

            # keep for future verbose wrapper logging
            # print("\nCurrent columns:", current_rewards_columns)
            # print("sorted columns:", sorted_rewards_columns)
            # print("Sorted df_rewards:", self.df_rewards.head(3))


    @class_verbose_logger
    # Update and keep the top n values 
    def update_worst_n_values(self, rewards:list[float], steps:list[int]):
        for index, _ in enumerate(self.worst_n_total_rewards):            
            if sum(rewards) < self.worst_n_total_rewards[-1]:
                self.worst_n_total_rewards.pop()
                self.worst_n_total_rewards.append(sum(rewards))
                self.df_worst_rewards.iloc[:, -1] = rewards
                self.df_worst_steps.iloc[:, -1] = steps

            sorted_indices = [
                index for index, _ 
                in sorted(enumerate(self.worst_n_total_rewards), key=lambda x: x[1])
            ]

            # re-sort the column based on sorted_indices
            current_rewards_columns = self.df_worst_rewards.columns
            current_steps_columns = self.df_worst_steps.columns
            sorted_rewards_columns = [
                current_rewards_columns[index] for index in sorted_indices
            ]
            sorted_steps_columns = [
                current_steps_columns[index] for index in sorted_indices
            ]

            self.df_worst_rewards = self.df_worst_rewards[sorted_rewards_columns]
            self.df_worst_steps = self.df_worst_steps[sorted_steps_columns]
            self.worst_n_total_rewards = [self.worst_n_total_rewards[index] for index in sorted_indices]

            # keep for future verbose wrapper logging
            # print("\nCurrent columns:", current_rewards_columns)
            # print("sorted columns:", sorted_rewards_columns)
            # print("Sorted df_rewards:", self.df_worst_rewards.head(3))


    @class_verbose_logger
    def train(self, data: np.array, verbose=True) -> tuple[np.array, dict]:
        environments_list = []
        total_rewards_list = []
        rewards_list = []
        steps_list = []

        # print("Begin training")

        for episode in range(self.num_train_episodes):

            # print("\nEpisode:", episode)

            # initialize cumulative rewards
            total_rewards = 0
            steps = []
            environments = []
            rewards = []

            # 10/8/2023: 
            # Not randomizing in starting because every state from now on is assume to 
            # start fresh without holding overnight
            current_state = (0,0,0)
            #print("In state:", current_state)

            # the recommended action will be choosen from the allowed actions listS
            action, q_value = self.policy(current_state)
            rewards_value = self.reward_function(current_state, action)

            steps.append(action)
            rewards.append(rewards_value)
            environments.append(current_state)

            for month in range(0, self.Q.shape[0]):
                for day in range(0, self.Q.shape[1]):
                    for time in range(0, self.Q.shape[2]):

                        # only modify the next state to start on 2nd state when the incoming state 
                        # is (0,0,0,0)
                        if current_state[:-1] == (0,0,0):
                            next_state = list(next_state)
                            next_state[2] = 1
                            next_state = tuple(next_state) # add 1 to time state if starting from (1,0,0,0) or (0,0,0,0)
                            #print("In modified next state:", next_state)
                        else:
                            next_state = tuple([month, day, time])
                            #print("In next state:", next_state)

                        action, _ = self.policy(current_state)
                        rewards_value = self.reward_function(current_state, action)
                        next_action, _ = self.policy(next_state)

                        self.update_q_value(current_state, action, rewards_value, next_state, next_action)

                        steps.append(action)
                        rewards.append(rewards_value)
                        environments.append(current_state)

                        current_state = next_state
                        action = next_action 


            # print(f"Episode {episode}:", len(rewards))
            # update to keep top n best action that give best rewards
            self.update_top_n_values(rewards=rewards, steps=steps)

            # update the worst n action to ensure it is really trying different actions 
            self.update_worst_n_values(rewards=rewards, steps=steps)

            # print(self.df_worst_rewards.head())
            # print(self.df_worst_steps.head())
    
            if episode % 100 == 0 and episode > 0:
                print("Backing up training model")
                self.save_model(episodes=episode)
                self.save_best_n_result(episodes=episode)

            if episode % 100 == 0 and episode > 0:
                print(f"Done {episode} episodes")

            # total_rewards_list.append(sum(rewards))
            # rewards_list.append(rewards)
            # steps_list.append(steps)
            # environments_list.append(environments)
        
        # self.train_value_dict['environments'] = environments_list 
        # self.train_value_dict['total_rewards'] = total_rewards_list
        # self.train_value_dict['rewards'] = rewards_list
        # self.train_value_dict['steps'] = steps_list

        #print("Finish training. All values are stored!")


    @class_verbose_logger
    def save_model(self, episodes=None):	
        if episodes is None:
            joblib.dump(self.Q, f'./validation/model/sarsa_crypto_{Utilator.get_formatted_date()}.joblib')
            joblib.dump(self.Q, f'./validation/model/sarsa_crypto.joblib')
        else:
            joblib.dump(self.Q, f'./validation/model/sarsa_crypto_backup.joblib')

        #print("Finish saving model and values dictionary!")


    @class_verbose_logger
    def save_best_n_result(self, episodes=None):
        if episodes is None:
            self.df_rewards.to_csv(f"./validation/result/top_{self.keep_top_n_steps}_rewards.csv")
            self.df_steps.to_csv(f"./validation/result/top_{self.keep_top_n_steps}_steps.csv")
            self.df_worst_rewards.to_csv(f"./validation/result/worst_{self.keep_top_n_steps}_rewards.csv")
            self.df_worst_steps.to_csv(f"./validation/result/worst_{self.keep_top_n_steps}_steps.csv")
        else: 
            self.df_rewards.to_csv(f"./validation/result/top_{self.keep_top_n_steps}_{episodes}_rewards.csv")
            self.df_steps.to_csv(f"./validation/result/top_{self.keep_top_n_steps}_{episodes}_steps.csv")
            self.df_worst_rewards.to_csv(f"./validation/result/worst_{self.keep_top_n_steps}_{episodes}_rewards.csv")
            self.df_worst_steps.to_csv(f"./validation/result/worst_{self.keep_top_n_steps}_{episodes}_steps.csv")        
        #print(f"Finish saving top {self.keep_top_n_steps} results as csv!")


    @class_verbose_logger
    def validate(self):
        # dataframe placeholder to store every iteration step, move, state for analysis
        df_validate_result = pd.DataFrame(
                                columns=[
                                    'environments',
                                    'total_rewards',
                                    'rewards',
                                    'steps'
                                ]
                            )	

        if self.Q is None:
            # read in the saved model 
            self.Q = joblib.load('./validation/model/sarsa_crypto_backup.joblib')
            print("Loaded in SARSA model!")
        
        if self.test_data is None: 
            self.test_data = pd.read_csv("./validation/data/test_data.csv")
            print("Loaded in test data!")

        df_preprocessed, _, _, _, _ = Preprocessor.preprocessing(self.test_data)
        self.price_table = Preprocessor.create_price_table(df_preprocessed, self.Q)

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

        # transform into state list for evaluation
        df_state['state_list'] = df_state.apply(lambda row: row.tolist(), axis=1)
        state_list = df_state['state_list']

        total_profit = 0
        single_profit = 0
        total_profit_list = []
        unit = 0
        budget = self.budget
        buy = 0
        sell = 0
        no_action = 0

        for state in state_list:
            state = tuple(state)
            action = np.argmax(self.Q[state])
            if action == 0 and budget > self.price_table[state]: # buy
                budget -= self.price_table[state]
                total_profit -= self.price_table[state]
                total_profit_list.append(total_profit)
                unit += (1*0.998)
                buy += 1
            elif action == 1 and unit > 0: # sell
                budget += unit * self.price_table[state] * 0.998
                total_profit += unit * self.price_table[state] * 0.998
                total_profit_list.append(total_profit)
                # print("Sell price:", self.price_table[state], "total_profit:", total_profit)
                unit = 0
                sell += 1

            elif action == 2: # no action
                no_action += 1
                pass
            else:
                if self.verbose:
                    if action == 0:
                        logger.debug(f"Couldnt perform action {Constantor.INDEX_TO_ACTION.get(action)} due to budget = {budget}")
                    else:
                        logger.debug(f"Couldnt perform action {Constantor.INDEX_TO_ACTION.get(action)} due to unit = {unit}")
                else:
                    pass

            if self.verbose:
                logger.debug(f"State: {state} Total profit: {total_profit} Action: {Constantor.INDEX_TO_ACTION.get(action)} Unit: {unit} Price: {self.price_table[state]} Budget: {budget}")

        potential_max_total_profit = [x for x in total_profit_list if x > 0]
        if self.verbose:
            logger.debug(f"Len potential max total_profit: {potential_max_total_profit}")
            if len(potential_max_total_profit) > 0:
                logger.debug(f"Average total_profit: {sum(potential_max_total_profit)/len(potential_max_total_profit)}")
            else:
                logger.debug("No total_profit")
            logger.debug(f"Unit: {unit} Budget: {budget}, Total Profit: {total_profit}")
            logger.debug(f"Buy counter: {buy} Sell counter: {sell} No action counter: {no_action}")
            