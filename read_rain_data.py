"""
Reads MIDAS rain data and returns values for given date/date range
"""
import numpy as np
import pandas as pd
import glob
import datetime
import pickle
from tqdm import tqdm


def construct_rain_dicts(csv_folder):
    """
    Constructs dictionary of yearly rain dataframes from csvs
    :param csv_folder: file path to rain data csvs
    :return:
    """
    # daily rain csvs
    all_daily_csvs = glob.iglob(csv_folder + "/Daily/*.csv")
    # hourly rain csvs
    all_hourly_csvs = glob.iglob(csv_folder + "/Hourly/*.csv")

    df_dict = {}

    for hourly_csv in all_hourly_csvs:
        hourly_df = pd.read_csv(hourly_csv, skiprows=61)
        hourly_df = hourly_df[hourly_df.ob_hour_count == 1]
        hourly_df = hourly_df[:-1]
        year = hourly_csv.split('.')[0][-4:]
        
        hourly_df['ob_end_time'] = pd.to_datetime(hourly_df['ob_end_time'])
        hourly_df = hourly_df.resample('D', on='ob_end_time', ).sum()

        # resamples hourly into daily
        ix = pd.date_range(datetime.date(int(year), 1, 1), datetime.date(int(year), 12, 31), freq='D')
        hourly_df = hourly_df.reindex(ix)
        hourly_df = hourly_df.reset_index(level=0).rename(columns={'index': 'ob_date'})
        
        df_dict[year] = hourly_df

    for daily_csv in all_daily_csvs:
        daily_df = pd.read_csv(daily_csv,skiprows=61)
        daily_df = daily_df[:-1]
        year = daily_csv.split('.')[0][-4:]
        
        daily_df['ob_date'] = pd.to_datetime(daily_df['ob_date'])
        daily_df = daily_df.set_index('ob_date')
        ix = pd.date_range(datetime.date(int(year), 1, 1), datetime.date(int(year), 12, 31), freq='D')
        daily_df = daily_df.reindex(ix)
        daily_df = daily_df.reset_index(level=0).rename(columns={'index': 'ob_date'})
        
        df_dict[year] = daily_df

    return df_dict


def get_rain_4_date(date, csv_folder,pkl = False):
    """
    Gets daily rain data for given date
    :param date: date
    :param csv_folder: file path to rain data csvs
    :param pkl: Bool, True -> gets dataframes dict from .pkl instead of csvs
    :return:
    """
    if not pkl:
        df_dict = construct_rain_dicts(csv_folder)
    else:
        df_dict = pickle.load(open("full_rain_dict.pkl", 'rb'))

    try:
        df = df_dict[str(date.year)]
        return df[pd.to_datetime(df['ob_date']).dt.date == date]['prcp_amt'].values[0]
    except KeyError:
        return np.NaN


def get_rain_4_date_range(start,end,csv_folder,pkl = False):
    """
    Gets rain data for date range [start,end]
    :param start: start of date range
    :param end: end of date range
    :param csv_folder: file path to rain data csvs
    :param pkl: Bool, True -> gets dataframes dict from .pkl instead of csvs
    :return:
    """
    if not pkl:
        df_dict = construct_rain_dicts(csv_folder)
    else:
        df_dict = pickle.load(open("full_rain_dict.pkl", 'rb'))

    if start.year == end.year:
        try:
            df = df_dict[str(start.year)]
            rain_arr = df[pd.to_datetime(df['ob_date']).dt.date.between(start,end)]['prcp_amt'].to_numpy()
        except KeyError:
            rain_arr = np.empty((end-start).days+1)
            rain_arr[:] = np.NaN
    else:
        yy = range(start.year, end.year + 1)
        rain_arr = np.empty(1)
        for y in yy:
            if y == start.year:
                y_start = start
                y_end = datetime.date(y, 12, 31)
            elif y == end.year:
                y_start = datetime.date(y, 1, 1)
                y_end = end
            else:
                y_start = datetime.date(y, 1, 1)
                y_end = datetime.date(y, 12, 31)

            n_days = (y_end - y_start).days + 1

            try:
                df = df_dict[str(y)]
                y_rain_df = df[pd.to_datetime(df['ob_date']).dt.date.between(y_start, y_end)]['prcp_amt']
                y_rain_arr = y_rain_df.to_numpy()
            except KeyError:
                y_rain_arr = np.empty(n_days)
                y_rain_arr[:] = np.NaN

            rain_arr = np.concatenate((rain_arr,y_rain_arr))
        rain_arr = rain_arr[1:]
    return rain_arr


def get_rain_4_date_pkl(date):
    """
    Gets rain data for date
    :param date:
    :return:
    """
    return get_rain_4_date(date,'',True)


def get_rain_4_date_range_pkl(start,end):
    """
    Gets rain data for dates in range [start, end]
    :param start:
    :param end:
    :return:
    """
    return get_rain_4_date_range(start,end,'',True)


def read_hourly_rain_full(file):
    """
    Gets hourly rain data dataframe from full MIDAS yearly .txt
    :param file: file path to .txt
    :return:
    """
    df = pd.read_csv(file,header = 0, low_memory= False,index_col=False,names = ['ob_end_time',
                                                                               'id',
                                                                               'id_type',
                                                                               'ob_hour_count',
                                                                               'version_num',
                                                                               'met_domain_name',
                                                                               'src_id',
                                                                               'rec_st_ind',
                                                                               'prcp_amt',
                                                                               'prcp_dur',
                                                                               'prcp_amt_q',
                                                                               'prcp_dur_q',
                                                                               'meto_stmp_time',
                                                                               'midas_stmp_etime',
                                                                               'prcp_amt_j'])


    df = df[df.src_id == 534]
    df = df[df.version_num == 1]
    df = df[df.ob_hour_count ==1][['ob_end_time', 'prcp_amt']]
    df['prcp_amt'] = pd.to_numeric(df['prcp_amt'],errors='coerce')
    return df


def read_daily_rain_full(file):
    """
    Gets daily rain data dataframe from full MIDAS yearly .txt
    :param file: file path to .txt
    :return:
    """
    df = pd.read_csv(file,header = 0,low_memory = False, index_col=False, on_bad_lines='skip',names = ['id',
                                                                  'id_type',
                                                                  'ob_date',
                                                                  'version_num',
                                                                  'met_domain_name',
                                                                  'ob_end_ctime',
                                                                  'ob_day_cnt',
                                                                  'src_id',
                                                                  'rec_st_ind',
                                                                  'prcp_amt',
                                                                  'ob_day_cnt_q',
                                                                  'prcp_amt_q',
                                                                  'meto_stmp_time',
                                                                  'midas_stmp_etime',
                                                                  'prcp_amt_j',
                                                               ])
    df = df[df.src_id == 534]
    df = df[df.version_num ==1]
    df = df[df.ob_day_cnt == 1][['ob_date','prcp_amt']]
    df['prcp_amt'] = pd.to_numeric(df['prcp_amt'])
    return df


def construct_rain_dicts_full(full_temp_path):
    """
    Constructs dict of yearly rain dataframes from full MIDAS data
    :param full_temp_path:
    :return:
    """
    all_daily_files = glob.iglob(full_temp_path + "/Full Daily/*.txt")
    all_hourly_files = glob.iglob(full_temp_path + "/Full Hourly/*.txt")

    df_dict = {}

    for daily_file, hourly_file in tqdm(zip(all_daily_files, all_hourly_files), total=61):

        year = daily_file.split('.')[0][-6:-2]

        if int(year) < 2001:
            df = read_daily_rain_full(daily_file)
            df['ob_date'] = pd.to_datetime(df['ob_date'])
            df = df.resample('D', on='ob_date', ).mean()
            ix = pd.date_range(datetime.date(int(year), 1, 1), datetime.date(int(year), 12, 31), freq='D')
            df = df.reindex(ix)
            df = df.reset_index(level=0).rename(columns={'index': 'ob_date'})

        else:
            df = read_hourly_rain_full(hourly_file)
            df['ob_end_time'] = pd.to_datetime(df['ob_end_time'])


            df = df.resample('D', on='ob_end_time', ).sum()

            ix = pd.date_range(datetime.date(int(year), 1, 1), datetime.date(int(year), 12, 31), freq='D')
            df = df.reindex(ix)
            df = df.reset_index(level=0).rename(columns={'index': 'ob_date'})
        df_dict[year] = df
    return df_dict


if __name__ == '__main__':
    rain_dict = construct_rain_dicts_full('Full Rain Data')
    pickle.dump(rain_dict, open("rain_dict.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    


    



