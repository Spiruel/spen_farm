"""
Reads MIDAS solar data and returns values for given date/date range
"""
import numpy as np
import pandas as pd
import glob
import datetime
import pickle
from tqdm import tqdm


def construct_sol_dict(csv_folder):
    """
   Constructs dictionary of yearly solar data dataframes from csvs
   :param csv_folder: file path to rain data csvs
   :return:
   """
    all_csvs = glob.iglob(csv_folder + "/*.csv")
    df_dict = {}

    for csv in all_csvs:
        year = csv.split('.')[0][-4:]

        df = pd.read_csv(csv, skiprows=75)
        df = df[df.ob_hour_count == 1]
        df = df[:-1]

        df['ob_end_time'] = pd.to_datetime(df['ob_end_time'])
        df = df.resample('D', on='ob_end_time', ).sum()

        ix = pd.date_range(datetime.date(int(year), 1, 1), datetime.date(int(year), 12, 31), freq='D')
        df = df.reindex(ix)
        df = df.reset_index(level=0).rename(columns={'index': 'ob_date'})

        df_dict[year] = df
    return df_dict


def get_sol_4_date(date, csv_folder,pkl = False):
    """
    Gets solar data for date
    :param date: date
    :param csv_folder: filepath to solar data csvs
    :param pkl: Bool, True -> get dataframe dict from .pkl instead of csvs
    :return:
    """
    if not pkl:
        df_dict = construct_sol_dict(csv_folder)
    else:
        df_dict = pickle.load(open("full_sol_dict.pkl", 'rb'))

    try:
        df = df_dict[str(date.year)]
        return df[pd.to_datetime(df['ob_date']).dt.date == date]['glbl_irad_amt'].values[0]
    except KeyError:
        return np.NaN


def get_sol_4_date_range(start, end, csv_folder,pkl = False):
    """
    Gets solar data for date range
    :param start: start of date range
    :param end: end of data range
    :param csv_folder: filepath to solar data csvs
    :param pkl: Bool, True -> get dataframe dict from .pkl instead of csvs
    :return:
    """
    if not pkl:
        df_dict = construct_sol_dict(csv_folder)
    else:
        df_dict = pickle.load(open("full_sol_dict.pkl", 'rb'))

    if start.year == end.year:
        try:
            df = df_dict[str(start.year)]
            sol_arr = df[pd.to_datetime(df['ob_date']).dt.date.between(start,end)]['glbl_irad_amt'].to_numpy()
        except KeyError:
            sol_arr = np.empty((end-start).days+1)
            sol_arr[:] = np.NaN
    else:
        yy = range(start.year, end.year + 1)
        sol_arr = np.empty(1)
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
                y_rain_df = df[pd.to_datetime(df['ob_date']).dt.date.between(y_start, y_end)]['glbl_irad_amt']
                y_sol_arr = y_rain_df.to_numpy()
            except KeyError:
                y_sol_arr = np.empty(n_days)
                y_sol_arr[:] = np.NaN

            sol_arr = np.concatenate((sol_arr,y_sol_arr))
        sol_arr = sol_arr[1:]
    return sol_arr


def get_sol_4_date_pkl(date):
    """
    Gets solar data for date from .pkl
    :param date:
    :return:
    """
    return get_sol_4_date(date,'',True)


def get_sol_4_date_range_pkl(start,end):
    """
    Gets solar data for date from .pkl
    :param start:
    :param end:
    :return:
    """
    return get_sol_4_date_range(start,end,'',True)


def read_sol_data_full(file):
    """
    Gets solar data dataframe from full MIDAS yearly .txt
    :param file: file path to .txt
    :return:
    """
    df = pd.read_csv(file,header = 0,low_memory=False,names = ['id',
                                                                                 'id_type',
                                                                                 'ob_end_time',
                                                                                 'ob_hour_count',
                                                                                 'version_num',
                                                                                 'met_domain_name',
                                                                                 'src_id',
                                                                                 'rec_st_ind',
                                                                                 'glbl_irad_amt',
                                                                                 'difu_irad_amt',
                                                                                 'glbl_irad_amt_q',
                                                                                 'difu_irad_amt_q',
                                                                                 'meto_stmp_time',
                                                                                 'midas_stmp_etime',
                                                                                 'direct_irad',
                                                                                 'irad_bal_amt',
                                                                                 'glbl_s_lat_irad_amt',
                                                                                 'glbl_horz,ilmn',
                                                                                 'direct_irad_q',
                                                                                 'irad_bal_amt_q',
                                                                                 'glbl_s_lat_irad_amt_q',
                                                                                 'glbl_horz,ilmn_q'])

    df = df[df.src_id == 534]
    df = df[df.version_num == 1]
    df = df[df.ob_hour_count == 1][['ob_end_time', 'glbl_irad_amt']]
    df['glbl_irad_amt'] = pd.to_numeric(df['glbl_irad_amt'], errors='coerce')
    return df


def construct_sol_dict_full(folder):
    """
    Constructs dict of yearly solar data dataframes from full MIDAS data
    :param folder: path to folder containing MIDAS solar .txts
    :return:
    """
    all_files = glob.iglob(folder + "/*.txt")
    df_dict = {}

    for file in tqdm(all_files,total = 21):
        year = file.split('.')[0][-6:-2]
        df = read_sol_data_full(file)

        df['ob_end_time'] = pd.to_datetime(df['ob_end_time'])
        df = df.resample('D', on='ob_end_time', ).sum()

        ix = pd.date_range(datetime.date(int(year), 1, 1), datetime.date(int(year), 12, 31), freq='D')
        df = df.reindex(ix)
        df = df.reset_index(level=0).rename(columns={'index': 'ob_date'})

        df_dict[year] = df
    return df_dict


if __name__=='__main__':
    sol_dict = construct_sol_dict_full('Full Solar Data')
    pickle.dump(sol_dict, open("sol_dict.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

