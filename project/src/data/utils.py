from datetime import datetime
import sys 

def get_formatted_date():
    current_date = datetime.now()
    formatted_date = current_date.strftime('%Y%m%d')
    return formatted_date


def log_print(*args, **kwargs):
    with open('print_log.txt', 'a') as f:
        print(*args, **kwargs, file=f)