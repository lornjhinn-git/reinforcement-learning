import src.sarsa as Sarsa

import pandas as pd 
import random

sarsa_tester = Sarsa.SARSA()

def test_update_top_n_values():
    test_episode = sarsa_tester.keep_top_n_steps
    test_values = 288 # the total encoded time
    for episode in range(test_episode):
        random_steps_values = [random.choice([-1, 0, 1]) for _ in range(test_values)]
        random_rewards_values = [random.choice([-1, 0, 1]) for _ in range(test_values)]
        print("random_steps_values", len(random_steps_values), random_steps_values)
        print("random reward values", len(random_rewards_values), random_rewards_values, "\n")
        sarsa_tester.update_top_n_values(random_rewards_values, random_steps_values)
        print("df_steps:", sarsa_tester.df_steps, "\n")
        print("df_rewards:", sarsa_tester.df_rewards, "\n")