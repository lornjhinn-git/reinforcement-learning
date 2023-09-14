from src.preprocessing import preprocessing as Preprocessor
import pandas as pd 
import numpy as np
from datetime import datetime, timedelta
from pandas.testing import assert_frame_equal
from src import agent as Agent


print("Begin testing")
df_train = pd.read_csv("validation/data/train_data.csv")
df_test = pd.read_csv("validation/data/test_data.csv")
df_train['datetime'] = df_train['datetime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
df_test['datetime'] = df_test['datetime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))


def test_get_day_of_week():

    global df_train, df_test

    df_days = pd.DataFrame({'local_numeric_day': [1,2,3,4,5,6,7], 'local_day':['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']})
    cols = ['local_numeric_day', 'local_day']
    try:
        _df_train = Preprocessor.get_day_of_week(df_train).sort_values(by='local_numeric_day')[cols].drop_duplicates().reset_index(drop=True)
        _df_test = Preprocessor.get_day_of_week(df_test).sort_values(by='local_numeric_day')[cols].drop_duplicates().reset_index(drop=True)
        
        print(f"The two dataframes values are equal. {_df_train.shape}, {_df_test.shape}")
        print(_df_train.head(10))
        print(_df_test.head(10))

        assert_frame_equal(_df_train, _df_test, check_index_type=False)
    
    except AttributeError: # str type 'datetime' column doesnt have datetime built-in function
        try:
            print("Detected attribute error")
            _df_train['datetime'] = df_train['datetime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
            _df_test['datetime'] = df_test['datetime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
            print("Converted from str to datetime format")

            _df_train = Preprocessor.get_day_of_week(df_train).sort_values(by='local_numeric_day')[cols].drop_duplicates().reset_index(drop=True)
            _df_test = Preprocessor.get_day_of_week(df_test).sort_values(by='local_numeric_day')[cols].drop_duplicates().reset_index(drop=True)
            
            print(f"In exception: Two dataframes values are equal. {_df_train.shape}, {_df_test.shape}")
            print(_df_train.head(10))
            print(_df_test.head(10))

            assert_frame_equal(_df_train, _df_test, check_index_type=False)
        except Exception as e:
            print(f"Error message: {e}") 


def test_encode_time():

    global df_train, df_test
    cols = ['encoded_time']

    try:
        _df_train = Preprocessor.encode_time(df_train)[cols].sort_values(by=cols).drop_duplicates().reset_index(drop=True)
        _df_test = Preprocessor.encode_time(df_test)[cols].sort_values(by=cols).drop_duplicates().reset_index(drop=True)
        assert_frame_equal(_df_train, _df_test, check_index_type=False)
        print(f"The two dataframes values are equal. {_df_train.shape}, {_df_test.shape}")
        print(_df_train.head(10))
        print(_df_test.head(10))
    except Exception as e:
        print(f"Error message: {e}") 


def test_pareto_distribution_bins():

    global df_train, df_test
    cols = ['volume_bins','amount']

    try:
        _df_train = Preprocessor.pareto_distribution_bins(df_train)[cols].sort_values(by=cols).drop_duplicates(subset=['volume_bins'], keep='first').reset_index(drop=True)
        _df_test = Preprocessor.pareto_distribution_bins(df_test)[cols].sort_values(by=cols).drop_duplicates(subset=['volume_bins'], keep='first').reset_index(drop=True)
        assert_frame_equal(_df_train, _df_test, check_index_type=False)
        print(f"The two dataframes values are equal. {_df_train.shape}, {_df_test.shape}")
        print(_df_train.head(10))
        print(_df_test.head(10))
    except Exception as e:
        print(f"Error message: {e}") 


def test_convert_str_to_datetime():
    input_str = '2023-07-19 15:30:00'
    converted_value = Preprocessor.convert_str_to_datetime(input_str)
    print(f"Original input: {input_str}")
    print(f"Converted value: {converted_value}")
    print(f"Type of converted value: {type(converted_value)}")


def test_create_price_table():
    # get the unique value of each column for each state 
    from sklearn.preprocessing import LabelEncoder 
    from datetime import timedelta
    encoder = LabelEncoder()

    ###### Parameters #########
    day_range = 14
    df, _, _, _, _ = Preprocessor.preprocessing(df_train)
    ###########################


    # additional for testing: manually converting date from object to datetime format for faster testing
    df['date'] = df['datetime'].apply(lambda x: x.date())


    ########## LOGIC ########################
    price_array = np.zeros((5,7,288,1))
    df['label_encoded_time'] = encoder.fit_transform(df[['encoded_time']])
    if day_range is None:
        df = df[['week_of_month', 'local_numeric_day', 'label_encoded_time', 'average_period_price']]\
            .groupby(['week_of_month', 'local_numeric_day', 'label_encoded_time'])\
            .mean().reset_index()
    else:
        start_date = df['date'].max() - timedelta(days=day_range)
        print("Max date:", df['datetime'].max())
        print("Start date:", start_date)
        df = df[df['date'] >= start_date]
        df = df[['week_of_month', 'local_numeric_day', 'label_encoded_time', 'average_period_price']]\
            .groupby(['week_of_month', 'local_numeric_day', 'label_encoded_time'])\
            .mean().reset_index()
        
    # Iterate over the rows of the DataFrame
    for index, row in df.iterrows():
        week_index = int(row['week_of_month'])
        day_index = int(row['local_numeric_day'])
        time_index = int(row['label_encoded_time'])
        value = row['average_period_price']
        if value == 0: print(week_index, day_index, time_index) 
        price_array[week_index, day_index, time_index] = value

    return price_array

    #############################################