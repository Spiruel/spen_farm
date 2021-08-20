import numpy as np
import pandas as pd
import glob
import datetime


def get_cosmos_df(csv):
    df = pd.read_csv(csv)
    df = df[1:]
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])

    df = df.set_index('DATE_TIME')
    ix = pd.date_range(datetime.date(int(2016), 1, 1), datetime.date(int(2019), 12, 31), freq='D')
    df = df.reindex(ix)
    df = df.reset_index(level=0).rename(columns={'index': 'DATE_TIME'})
    return df


def get_cosmos_col_4_date(col,date,csv):
    df = get_cosmos_df(csv)
    try:
        val = df[df['DATE_TIME'].dt.date == date][col].values[0]
        if val == -9999:
            return np.NaN
        else:
            return val
    except IndexError:
        return np.NaN


def get_cosmos_col_4_date_range(col,start,end,csv):
    df = get_cosmos_df(csv)

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

if __name__ == '__main__':
    csv = "COSMOS-UK_SPENF_HydroSoil_Daily_2013-2019.csv"


    date = datetime.date(3000,8,1)

    start = datetime.date(2016,11,22)
    end = datetime.date(2017,1,1)
    #print(get_cosmos_col_4_date('COSMOS_VWC',date,csv))
    #print(get_cosmos_col_4_date('ALBEDO', date, csv))
    print(get_cosmos_col_4_date_range('ALBEDO',start,end,csv))