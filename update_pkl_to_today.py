import numpy as np
import pandas as pd
import datetime
import pickle


def most_recent_date(df):
    return df.dropna().ob_date.max()


if __name__ == '__main__':
    (soil_df_dict, air_df_dict) = pickle.load(open("full_temperature_dict.pkl", 'rb'))
    rain_df_dict = pickle.load(open("full_rain_dict.pkl", 'rb'))
    sol_df_dict = pickle.load(open("full_sol_dict.pkl", 'rb'))

    today = datetime.date.today()
    rain_21_df = rain_df_dict['2021']

    print(most_recent_date(rain_21_df))
