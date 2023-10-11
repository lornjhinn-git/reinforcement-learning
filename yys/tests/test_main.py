from ..main import * 
from ..src.config import config as Config


verbose = False

def test_find_next_value():
    for game_status in ['win', 'lose']:
        for mindset in ['LM', 'WM', 'SM']:
            next_value = find_next_value(game_status, mindset, Config.rps_mindset_action_value_dictionary)
            if verbose:
                print(game_status, mindset, next_value)


def test_find_next_action():
    for action in ['r', 'p', 's']:
        for mindset in ['LM', 'WM', 'SM']:
            if mindset == 'LM':
                target_value = 1
            if mindset in ['WM','SM']:
                target_value = 0
            next_action = find_next_action(action, target_value)
            if verbose:
                print(mindset, target_value, action, next_action)


def test_get_game_result():
    # verbose=True
    get_game_result(1, -1, verbose)
    get_game_result(0, 0, verbose)
    get_game_result(-1, 1, verbose)


def test_get_player_next_action():
    # verbose = True
    get_player_next_action(1,1,'r','LM', verbose) # win 
    get_player_next_action(1,2,'r','WM', verbose) # lose  
    get_player_next_action(None,None,'r','LM', verbose) # tie


def test_game_result():
    # verbose = True
    get_game_result(1, -1, verbose)
    get_game_result(0, 0, verbose)
    get_game_result(-1, 1, verbose)


def test_play():
    verbose = True
    episode = 10
    play(episode, verbose)
