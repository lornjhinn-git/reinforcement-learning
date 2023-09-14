from datetime import datetime
import sys 
import pickle 
import os 
import uuid 


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


def create_model_id():
    # Generate a UUID
    unique_id = uuid.uuid4()

    # Get the current date and time
    current_datetime = datetime.now()

    # Format the date and time as a string in a timestamp-like manner (YYYYMMDDHHMMSS)
    timestamp_str = current_datetime.strftime("%Y%m%d%H%M%S")

    # Combine the UUID and timestamp to create a unique ID
    combined_id = f"{timestamp_str}_{unique_id}"

    return combined_id