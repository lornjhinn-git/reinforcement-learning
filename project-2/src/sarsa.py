from .constants import constants as Constantor
from .preprocessing import utils as Utilator
from .preprocessing import preprocessing as Preprocessor

from typing import Optional
import pandas as pd
import numpy as np
import random 
import joblib


class SARSA:
    def __init__(
            self, 
            learning_rate=0.005,
            epsilon=0.1,
            gamma=0.9,
            num_train_episodes=100,
            num_test_episodes=1,
            data=None,
            Q=None,
            reward_table=None,
            budget=2000,
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
            keep_top_n_steps=None
    ):
        self.learning_rate=0.005 if learning_rate is None else learning_rate
        self.epsilon=0.1 if epsilon is None else epsilon
        self.gamma=0.9 if gamma is None else gamma
        self.num_train_episodes=100 if num_train_episodes is None else num_train_episodes
        self.num_test_episodes=1 if num_test_episodes is None else num_test_episodes
        self.data=data
        self.Q=Q
        self.reward_table=reward_table
        self.budget=2000 if budget is None else budget  # set at 2k usd as starting budget
        self.price_table=price_table
        self.purchase_unit=0 if purchase_unit is None else purchase_unit
        self.total_purchase_prices=0 if total_purchase_prices is None else total_purchase_prices
        self.purchase_states=purchase_states
        self.train_data=train_data
        self.test_data=test_data
        self.train_value_dict=train_value_dict
        self.test_value_dict=test_value_dict
        self.keep_top_n_steps=10 if keep_top_n_steps is None else keep_top_n_steps
        self.df_steps=pd.DataFrame({f'Step_Top_{i+1}': [0]*288 for i in range(self.keep_top_n_steps)})
        self.df_rewards=pd.DataFrame({f'Rewards_Top_{i+1}': [0]*288 for i in range(self.keep_top_n_steps)})
        self.top_n_total_rewards = [0]*self.keep_top_n_steps



class SARSA(SARSA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Child sarsa initialized")
        print(self.__dict__)


    def policy(self, state) -> tuple[int, float]: 
        r_var = random.random()
        if r_var < self.epsilon:
            # print("random")
            best_action = np.where(self.Q[state] == random.choice(self.Q[state]))[0][0]
            best_value = self.Q[state][best_action]
        else:
            best_action = np.argmax(self.Q[state])
            best_value = self.Q[state][best_action]

        return best_action, best_value


    def reward_function(self, state, action) -> float:
        if (
            # selling profit > average purchase price + trading fees
            action == 1 and 
            self.purchase_unit > 0 and 
            self.price_table[state] > ((self.total_purchase_prices/len(self.purchase_unit)) + (self.price_table[state]*0.002))
        ): # making profit
            rewards_value = 1
        elif (
            (
                # selling profit <= average purchase price + trading fees
                action == 1 and 
                self.purchase_unit > 0 and 
                self.price_table[state] <= ((self.total_purchase_prices/len(self.purchase_unit)) + (self.price_table[state]*0.002))
            ) or 
            (
                # total purchase more than budget 
                action == 0 and 
                self.purchase_unit > 0 and 
                self.total_purchase_prices > self.budget
            ) or 
            (
                # try to sell even no unit
                action == 1 and 
                self.purchase_unit == 0
            ) 
        ): # losing money
            rewards_value = -1
        else: # no actionW
            rewards_value = 0

        if action == 0: # buy 
            self.purchase_unit += 1
            self.total_purchase_prices += self.price_table[state]
            self.budget -= self.price_table[state]
        elif action == 1 and self.purchase_unit > 0: # sell
            self.budget = self.price_table[state]*0.998
            self.purchase_prices = 0
            self.purchase_unit = 0

        return rewards_value


    # Update Q-value for a state-action pair based on observed rewards and estimated future Q-values
    def update_q_value(self, state:tuple, action:int, rewards_value:float, next_state:tuple, next_action:int):

        # Compute the updated Q-value using the self update equation
        current_q = self.Q[state][Constantor.ACTION_DICT.get(action)]
        next_q = self.Q[next_state][Constantor.ACTION_DICT.get(next_action)]
        new_q = current_q + self.learning_rate * (rewards_value + self.gamma * next_q - current_q)
        
        # Update the Q-value in the Q-table
        self.Q[state][Constantor.ACTION_DICT.get(action)] = new_q
    

    # Update and keep the top n values 
    def update_top_n_values(self, rewards:list[float], steps:list[int]):
        for index, _ in enumerate(self.top_n_total_rewards, start=1):            
            if sum(rewards) > self.top_n_total_rewards[-1]:
                print("Replacing")
                self.top_n_total_rewards.pop()
                self.top_n_total_rewards.append(sum(rewards))
                self.df_rewards.iloc[:, -1] = rewards
                self.df_steps.iloc[:, -1] = steps       

            self.top_n_total_rewards.sort(reverse=True)
            sorted_indices = [
                index for index, _ 
                in sorted(enumerate(self.top_n_total_rewards), key=lambda x: x[1], reverse=True)
            ]

            # re-sort df_steps, df_rewards based on the sorted_indices 
            self.df_steps = self.df_steps.iloc[:, sorted_indices]
            self.df_rewards = self.df_rewards.iloc[:, sorted_indices]


    def train(self, data: np.array, verbose=True) -> tuple[np.array, dict]:
        environments_list = []
        total_rewards_list = []
        rewards_list = []
        steps_list = []

        for episode in range(self.num_train_episodes):

            print("\nEpisode:", episode)

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

                        next_action, q_value = self.policy(next_state)

                        self.update_q_value(current_state, action, rewards_value, next_state, next_action)

                        current_state = next_state
                        action = next_action 
                        
                        steps.append(action)
                        rewards.append(rewards_value)
                        environments.append(current_state)




            # total_rewards_list.append(sum(rewards))
            # rewards_list.append(rewards)
            # steps_list.append(steps)
            # environments_list.append(environments)
        
        # self.train_value_dict['environments'] = environments_list 
        # self.train_value_dict['total_rewards'] = total_rewards_list
        # self.train_value_dict['rewards'] = rewards_list
        # self.train_value_dict['steps'] = steps_list

        print("Finish training. All values are stored in self!")


    def save_model(self):	
        joblib.dump(self.Q, f'sarsa_crypto.joblib')
        joblib.dump(self.Q, f'sarsa_crypto_{Utilator.get_formatted_date()}.joblib')

        # for key in self.train_value_dict:
        #     with open(f'{key}.txt', 'w') as file:
        #         for item in self.train_value_dict.get(key):
        #             file.write(str(item) + '\n')

        print("Finish saving model and values dictionary!")


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
            self.Q = joblib.load('self_crypto.joblib')
            print("Loaded in self model!")
        
        if self.test_data is None: 
            self.test_data = pd.read_csv("test_data.csv")
            print("Loaded in test data!")

        df_preprocessed_test = Preprocessor.preprocessing(self.test_data)

        # create the new state table from df_preprocessed test


        environments_list = []
        total_rewards_list = []
        rewards_list = []
        steps_list = []

        for episode in range(self.num_test_episodes):

            total_rewards = 0
            steps = []
            environments = []
            rewards = []

            current_state = (0,0,0)
            action = np.argmax(self.Q[current_state])
            rewards_value = self.reward_function[current_state][action]

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

                        next_action, q_value = self.policy(next_state)
                        rewards_value = self.reward_function[current_state][action]

                        current_state = next_state
                        action = next_action 
                        
                        steps.append(action)
                        rewards.append(rewards_value)
                        environments.append(current_state)

                total_rewards += sum(rewards)
                self.test_value_dict['environments'] = environments
                self.test_value_dict['total_rewards'] = total_rewards
                self.test_value_dict['rewards'] = rewards
                self.test_value_dict['steps'] = steps

            df_validate_result.loc[len(df_validate_result)] = self.test_value_dict

        average_rewards = total_rewards / self.num_test_episodes
        df_validate_result.to_csv(f'./validation/result_{Utilator.get_formatted_date()}.csv')
        print(f"Average reward on test data: {average_rewards}") 
        
        