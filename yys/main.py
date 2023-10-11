# LM
#   - win or lose: change the move that will lose to previous

# WM
#   - win : stick to the previous move 
#   - lose: change to any other move except previous move 

# SM
#   - win or lose: stick to the previous move


# game rules: 
#       r   p   s
#   r   0  -1   1
#   p   1   0  -1
#   s  -1   1   0
#

import numpy as np
import random
from .src.config import config as Config


def find_next_value(game_status:str, mindset:str, dictionary=Config.rps_mindset_action_value_dictionary) -> str:
    for key, value in dictionary.get(game_status).items():
        if key == mindset:
            return value
        if isinstance(value, dict):
            sub_key = find_next_value(game_status, mindset)
            if sub_key is not None:
                return key + '.' + sub_key
    return None


def find_next_action(current_action, target_value, dictionary=Config.rps_action_dictionary) -> str:
    for key, value in dictionary.get(current_action).items():
        if value == target_value:
            return key
        if isinstance(value, dict):
            sub_key = find_next_action(current_action, target_value)
            if sub_key is not None:
                return key + '.' + sub_key
    return None


def get_game_result(p1_value:int, p2_value:int, verbose=False) -> tuple[int, int, bool]:
    isTie = False
    winner = None
    loser = None

    if p1_value > p2_value:
        winner = 1
        loser = 2
    elif p1_value == p2_value:
        isTie = True
    else:
        winner = 2
        loser = 1

    if verbose:
        print(p1_value, p2_value, winner, loser, isTie)

    return winner, loser, isTie

# 11/10/2023: Currently only assume play with ourself despite what mindset we predict on opponent
# Optional enhancement: Probability to flip move based on predicted opponent mindset
def get_player_next_action(
        player: int,
        winner: int,
        player_action: str,
        player_mindset:str, 
        verbose = False,
        player_predict_opponent_mindset: str = None
) -> str:
    if winner is not None:
        if player == winner: 
            value = find_next_value('win', player_mindset)
        else:
            value = find_next_value('lose', player_mindset)
    else: # tie game 
        value = random.choice([-1, 0, 1])

    # if is list form else just default action
    if isinstance(value, list):
        value = random.choice(value)

    action = find_next_action(player_action, value)

    if verbose:
        print(player, winner, player_action, player_mindset, value, action)

    return action


def play(episodes=Config.EPISODES, verbose=False):
    counter = 0
    for p1_action in ['r', 'p', 's']:
        for p2_action in ['r', 'p', 's']:
            # p1_action = random.choice(['r','p', 's'])
            # p2_action = random.choice(['r','p', 's']) 
            p1_value = Config.rps_action_dictionary[p1_action][p2_action]
            p2_value = Config.rps_action_dictionary[p2_action][p1_action]
                
            winner, loser, isTie = get_game_result(p1_value, p2_value)

            for p1_mindset in Config.game_mindset:
                for p1_predict_p2_mindset in Config.game_mindset: 
                    for p2_mindset in Config.game_mindset:
                        for p2_predict_p1_mindset in Config.game_mindset: 

                            p1_action = get_player_next_action(1, winner, p1_action, p1_mindset) 
                            p2_action = get_player_next_action(2, winner, p2_action, p2_mindset) 

                            p1_value = Config.rps_action_dictionary[p1_action][p2_action]
                            p2_value = Config.rps_action_dictionary[p2_action][p1_action]

                            winner, loser, isTie = get_game_result(p1_value, p2_value)
                            counter += 1

                            if verbose:
                                print(p1_mindset + '_' + p1_predict_p2_mindset, p2_mindset + '_' + p2_predict_p1_mindset)
                                print('p1 mindset:', p1_mindset , 'p1_action:', p1_action,  'p2 mindset:', p2_mindset, 'p2 action:', p2_action, 'winner:', winner)
                                
            print("Total counter:", counter)



