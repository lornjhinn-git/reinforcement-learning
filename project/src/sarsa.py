from constants import constants as Constantor
from preprocessing import utils as Utilator


from typing import Optional
import pandas as pd
import numpy as np
import random 

class Parentself:
    def __init__(self):
        self.learning_rate: Optional[float] = 0.005
        self.discount_factor: Optional[float] = 0.1
        self.epsilon: Optional[float] = 0.1
        self.gamma: Optional[float] = 0.9
        self.num_train_episodes: Optional[int] = 10000
        self.num_test_episodes: Optional[int] = 100
        self.data: Optional[pd.Dataframe] = None
        self.Q: Optional[np.array] = None
        self.budget: float = 2000 # set at 2k usd as starting budget
        self.price_table: np.arary = None
        self.purchase_unit: int = 0
        self.purchase_prices: list[float] = None
        self.purchase_states: list[tuple] = None
        self.train_data: Optional[pd.Dataframe] = None
        self.test_data: Optional[pd.Dataframe] = None
        self.train_value_dict = {
                'environments': None,
                'total_rewards': None,
                'rewards': None, 
                'steps': None
	    }
        self.test_value_dict = {
                'environments': None,
                'total_rewards': None,
                'rewards': None, 
                'steps': None
	    }


    def policy(self, state) -> tuple[int, float]: 
        r_var = random.random()
        if r_var < self.epsilon:
            print("random")
            best_action = np.where(self.Q[state] == random.choice(self.Q[state]))[0][0]
            best_value = self.Q[state][best_action]
        else:
            best_action = np.argmax(self.Q[state])
            best_value = self.Q[state][best_action]

        return best_action, best_value


    def reward_function(self, state, action, next_state) -> float:

        if action == 0: # buy 
            # update self attribute
            self.purchase_unit += 1
            self.purchase_prices.append(self.price_table[state])

        return rewards_value


    # Update Q-value for a state-action pair based on observed rewards and estimated future Q-values
    def update_q_value(self, state:tuple, action:int, rewards:list, rewards_value:float, next_state:tuple, next_action:int, verbose=False):

        # Compute the updated Q-value using the self update equation
        current_q = self.Q[state][Constantor.ACTION_DICT.get(action)]

        # Additional reward if have been making profit of at least 20 usd
        if sum(rewards) >= 20: current_q += 100
        next_q = self.Q[next_state][Constantor.ACTION_DICT.get(next_action)]
        new_q = current_q + self.learning_rate * (rewards_value + self.gamma * next_q - current_q)
        
        # Update the Q-value in the Q-table
        self.Q[state][Constantor.ACTION_DICT.get(action)] = new_q
        
        # Check if the (state, action) pair exists in the Q-table
        # if (state, action) not in Q:
        #     Q[(state, action)] = 0.0


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
            current_state = tuple(self.Q.shape[:-1])

            # the recommended action will be choosen from the allowed actions listS
            action, q_value = policy(current_state)

            steps.append(action)
            rewards.append(rewards_value)
            environments.append(current_state)

            # dummy initialization for next state
            next_state = current_state

            for month in range(0, self.reward_table.shape[1]):
                for day in range(0, self.reward_table.shape[2]):
                    for time in range(0, self.reward_table.shape[3]):

                        # only modify the next state to start on 2nd state when the incoming state 
                        # is (0,0,0,0) or (1,0,0,0)
                        if current_state[1:] == (0,0,0):
                            next_state = list(next_state)
                            next_state[3] = 1
                            next_state = tuple(next_state) # add 1 to time state if starting from (1,0,0,0) or (0,0,0,0)
                        else:
                            next_state = tuple([next_state[0], month, day, time])

                        # second layer modification on state based on the current action
                        if action == 0: # buy, next state become holding state
                            next_state = (1,) + (month, day, time)
                        elif action == 1: # sell, next state become not holding state 
                            next_state = (0,) + (month, day, time)
                        else:  # if no action, then stick to the current holding state
                            # print("No action")
                            pass

                        next_action, next_action_value = policy(next_state)


                        update_q_value(current_state, action, rewards, rewards_value, next_state, next_action, verbose=False)

                        steps.append(action)
                        rewards.append(rewards_value)
                        environments.append(current_state)

                        current_state = next_state
                        action = next_action 
                        action_value = next_action_value

            total_rewards_list.append(sum(rewards))
            rewards_list.append(rewards)
            steps_list.append(steps)
            environments_list.append(environments)
        
        self.train_value_dict['environments'] = environments_list 
        self.train_value_dict['total_rewards'] = total_rewards_list
        self.train_value_dict['rewards'] = rewards_list
        self.train_value_dict['steps'] = steps_list

        print("Finish training. All values are stored in self!")


    def save_model(self):	
        joblib.dump(self.Q, f'self_crypto.joblib')
        joblib.dump(self.Q, f'self_crypto_{Utilator.get_formatted_date()}.joblib')

        for key in self.train_value_dict:
            with open(f'{key}.txt', 'w') as file:
                for item in self.train_value_dict.get(key):
                    file.write(str(item) + '\n')

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

        environments_list = []
        total_rewards_list = []
        rewards_list = []
        steps_list = []

        for episode in range(self.num_test_episodes):

            total_rewards = 0
            steps = []
            environments = []
            rewards = []

            # hard assign the starting holding state, can determine by ourselves to start with holding or not holding
            # currently set to not holding as dont want to hold overnight while sleeping and not monitoring 
            holding_state = 0

            # 28/07/2023: currently the validation will always start from 1/0 + (0,0,0), which mean always starting from a new day new time
            # in the future can add in the enhanced version where it start from where the train data last 
            # Note: the reward table here is the validation reward table 
            current_state = (holding_state,0,0,0)
            action = np.argmax(self.Q[current_state])
            rewards_value = self.reward_table[current_state][action]

            steps.append(action)
            rewards.append(rewards_value)
            environments.append(current_state)

            # dummy initialization for next state
            next_state = current_state

            for month in range(0, self.reward_table.shape[1]):
                for day in range(0, self.reward_table.shape[2]):
                    for time in range(0, self.reward_table.shape[3]):
                        if current_state[1:] == (0,0,0):
                            next_state = list(next_state)
                            next_state[3] = 1
                            next_state = tuple(next_state) # add 1 to time state if starting from (1,0,0,0) or (0,0,0,0)
                        else:
                            next_state = tuple([next_state[0], month, day, time])
                    
                        # second layer modification on state based on the current action
                        if action == 0: # buy, next state become holding state
                            next_state = (1,) + (month, day, time)
                        elif action == 1: # sell, next state become not holding state 
                            next_state = (0,) + (month, day, time)
                        else:  # if no action, then stick to the current holding state
                            # print("No action")
                            pass

                        if next_state[0] == 1: # if in holding state, then only can sell or no action
                            allowed_next_actions = ['sell', 'no_action']
                        else:
                            allowed_next_actions = ['buy', 'no_action']

                        allowed_next_action_indices = [Constantor.ACTION_DICT[action] for action in allowed_next_actions]
                        allowed_next_action_q_values = self.Q[next_state][allowed_next_action_indices]

                        print("Allowed next action q values:", allowed_next_action_q_values)

                        next_action = np.argmax([allowed_next_action_q_values])

                        print("Next action:", next_action)

                        current_state = next_state
                        action = next_action
                        rewards_value = self.reward_table[current_state][action]

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
        