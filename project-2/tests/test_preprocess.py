from src.data import preprocess as Preprocessor
import pandas as pd 
import numpy as np
from datetime import datetime
from pandas.testing import assert_frame_equal


print("Begin testing")
df_train = pd.read_csv("train_data.csv")
df_test = pd.read_csv("test_data.csv")
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


