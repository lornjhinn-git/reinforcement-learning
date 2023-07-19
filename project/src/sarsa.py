from typing import Optional
import pandas as pd
import numpy as np

class ParentSARSA:
    def __init__(self, verbose=True):
        self.learning_rate: Optional[float] = 0.005
        self.discount_factor: Optional[float] = 0.1
        self.epsilon: Optional[float] = 0.1
        self.gamma: Optional[float] = 0.9
        self.num_episodes: Optional[int] = 100000
        self.data: Optional[pd.Dataframe] = None
        self.Q: Optional[np.array] = None
        self.reward_table: Optional[np.array] = None
        self.isHolding = False

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
        