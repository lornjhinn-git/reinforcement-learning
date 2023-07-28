from typing import Optional
import pandas as pd
import numpy as np

class ParentSARSA:
    def __init__(self, verbose=True):
        self.learning_rate: Optional[float] = 0.005
        self.discount_factor: Optional[float] = 0.1
        self.epsilon: Optional[float] = 0.1
        self.gamma: Optional[float] = 0.9
        self.num_train_episodes: Optional[int] = 1000000
        self.num_test_episodes: Optional[int] = 10
        self.data: Optional[pd.Dataframe] = None
        self.Q: Optional[np.array] = None
        self.reward_table: Optional[np.array] = None
        self.isHolding = False
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

        if verbose:
            print("Parent SARSA inherited!")
            print("Parent attributes:", self.__dict__)


class SARSA(ParentSARSA): 
    def __init__(self, verbose=True, **kwargs):
        super().__init__(verbose=verbose)
        for key, value in kwargs.items():
            setattr(self, key, value)

        if verbose:
            print("SARSA child initialized")
            print("Child attributes:", self.__dict__)

    
    # def set_Q_table(self, data:pd.DataFrame):
        