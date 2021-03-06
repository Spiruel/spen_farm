"""
Functions to read COSMOS data csv and get data for given date/date range
"""
import numpy as np
import pandas as pd
import datetime
import pickle


def get_cosmos_df(csv):
    """
    Reads COSMOS data csv into dataframe
    :param csv: filepath to csv
    :return:
    """
    names = pd.read_csv(csv, skiprows=3, nrows=0)
    names = names.rename(columns={'parameter-id': 'DATE_TIME', 'COSMOS_VWC_1DAY': 'COSMOS_VWC',
                                  'ALBEDO_DAILY_MEAN': 'ALBEDO'}).columns.values

    df = pd.read_csv(csv, skiprows=5, header=0, index_col=False, dayfirst=True, names=names)
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], dayfirst=True)
    df = df.set_index('DATE_TIME')

    ix = pd.date_range(datetime.date(int(2016), 1, 1), datetime.date(int(2021), 12, 31), freq='D')
    df = df.reindex(ix)
    df = df.reset_index(level=0).rename(columns={'index': 'DATE_TIME'})
    return df


def get_cosmos_col_4_date(col,date,csv,pkl = False):
    """
    Gets the value of a column from COSMOS csv for a given date
    :param col: column to get values for
    :param date: date to get values for
    :param csv: COSMOS csv file path
    :param pkl: Bool, if true gets df from .pkl instead of csv
    :return:
    """
    if not pkl:
        df = get_cosmos_df(csv)
    else:
        df = pickle.load(open("cosmos_df.pkl", 'rb'))
    try:
        val = df[df['DATE_TIME'].dt.date == date][col].values[0]
        if val == -9999:
            return np.NaN
        else:
            return val
    except IndexError:
        return np.NaN


def get_cosmos_col_4_date_range(col,start,end,csv,pkl = False):
    """
    Gets the value of a column from COSMOS csv for a given range of dates
    :param col: column to get values for
    :param start: start of date range
    :param end: end of date range
    :param csv: COSMOS csv file path
    :param pkl: Bool, if true gets df from .pkl instead of csv
    :return:
    """
    if not pkl:
        df = get_cosmos_df(csv)
    else:
        df = pickle.load(open("cosmos_df.pkl", 'rb'))

    # if date range starts and ends in the same year easy to get values
    if start.year == end.year:

        cos_arr = df[df['DATE_TIME'].dt.date.between(start,end)][col].to_numpy()
        cos_arr[cos_arr == -9999] = np.NaN
        if cos_arr.size == 0:
            cos_arr = np.empty((end-start).days+1)
            cos_arr[:] = np.NaN

    else:
        yy = range(start.year, end.year + 1)
        cos_arr = np.empty(1)
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

            y_cos_arr = df[df['DATE_TIME'].dt.date.between(y_start,y_end)][col].to_numpy()
            y_cos_arr[y_cos_arr == -9999] = np.NaN
            if y_cos_arr.size == 0:
                y_cos_arr = np.empty(n_days)
                y_cos_arr[:] = np.NaN

            cos_arr = np.concatenate((cos_arr, y_cos_arr))
        cos_arr = cos_arr[1:]

    return cos_arr


def get_cosmos_col_4_date_pkl(col,date):
    """
    Gets value of col on date using .pkl
    :param col: column to get value for
    :param date: daet to get value for
    :return:
    """
    return get_cosmos_col_4_date(col,date,'',True)


def get_cosmos_col_4_date_range_pkl(col,start,end):
    """
    Gets value of col for date range
    :param col: col to get value for
    :param start: start of date range
    :param end: end of date range
    :return:
    """
    return get_cosmos_col_4_date_range(col,start,end,'',True)


if __name__ == '__main__':
    csv = "SPENF-2016-11-24-2021-09-05.csv"
    df = get_cosmos_df(csv)
    pickle.dump(df, open("cosmos_df.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
