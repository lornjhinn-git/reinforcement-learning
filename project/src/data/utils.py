from datetime import datetime

def get_formatted_date():
    current_date = datetime.now()
    formatted_date = current_date.strftime('%Y%m%d')
    return formatted_date