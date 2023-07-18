from typing import Optional
import pandas as pd
import numpy as np

class SARSA:
    def __init__(self, verbose=False):
        learning_rate: Optional[float] = 0.005
        discount_factor: Optional[float] = 0.1
        epsilon: Optional[float] = 0.1
        data: Optional[pd.Dataframe] = None
        q_table: Optional[np.array] = None
        reward_table: Optional[np.array] = None

        if verbose:
            print("Parent SARSA inherited!")
            print("Parent attributes:", self.__dict__)


class SARSA(SARSA): 
    def __init__(self, verbose=False, **kwargs):
        super().__init__(verbose=verbose)
        for key, value in kwargs.items():
            setattr(self, key, value)

        if verbose:
            print("SARSA child initialized")
            print("Parent attributes:", self.__dict__)
