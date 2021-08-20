import numpy as np
import pandas as pd
import glob
import datetime


def construct_sol_dict(csv_folder):
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

def get_sol_4_date(date, csv_folder):
    df_dict = construct_sol_dict(csv_folder)
    try:
        df = df_dict[str(date.year)]
        return df[pd.to_datetime(df['ob_date']).dt.date == date]['glbl_irad_amt'].values[0]
    except KeyError:
        return np.NaN



def get_sol_4_date_range(start, end, csv_folder):
    df_dict = construct_sol_dict(csv_folder)
    
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
    
    
    
if __name__=='__main__':
    date = datetime.date(2019,1,1)
    date2 = datetime.date(2018,1,1)


