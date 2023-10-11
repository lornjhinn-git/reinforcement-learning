EPISODES = 100

game_mindset = ['LM', 'WM', 'SM']

rps_mindset_action_value_dictionary = {
    'win': {
        'LM': 1,
        'WM': 0,
        'SM': 0
    },
    'lose': {
        'LM': 1,
        'WM': [1,-1],
        'SM': 0
    }
}

rps_action_dictionary = {
    'r': {'r': 0, 'p': -1, 's': 1}, 
    'p': {'r': 1, 'p': 0, 's': -1}, 
    's': {'r': -1, 'p': 1, 's': 0}
}