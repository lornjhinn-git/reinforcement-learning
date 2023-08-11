from datetime import datetime
import sys 
import pickle 
import os 

def get_formatted_date():
    current_date = datetime.now()
    formatted_date = current_date.strftime('%Y%m%d')
    return formatted_date


def log_print(*args, **kwargs):
    with open('print_log.txt', 'a') as f:
        print(*args, **kwargs, file=f)


def save_dictionary(dictionary:dict, dictionary_name:str, save_path:str='./validation'):
        with open(os.path.join(save_path, f'{dictionary_name}.pkl'), 'wb') as f:
             pickle.dump(dictionary, f)