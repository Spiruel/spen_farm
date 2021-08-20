import numpy as np
import pandas as pd
import glob
import datetime
import pickle

def construct_rain_dicts(csv_folder):
    all_daily_csvs = glob.iglob(csv_folder + "/Daily/*.csv")
    all_hourly_csvs = glob.iglob(csv_folder + "/Hourly/*.csv")

    df_dict = {}

    for hourly_csv in all_hourly_csvs:
        hourly_df = pd.read_csv(hourly_csv, skiprows=61)
        hourly_df = hourly_df[hourly_df.ob_hour_count == 1]
        hourly_df = hourly_df[:-1]
        year = hourly_csv.split('.')[0][-4:]
        
        hourly_df['ob_end_time'] = pd.to_datetime(hourly_df['ob_end_time'])
        hourly_df = hourly_df.resample('D', on='ob_end_time', ).sum()

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


def get_rain_4_date(date, csv_folder):
    df_dict = construct_rain_dicts(csv_folder)
    try:
        df = df_dict[str(date.year)]
        return df[pd.to_datetime(df['ob_date']).dt.date == date]['prcp_amt'].values[0]
    except KeyError:
        return np.NaN


def get_rain_4_date_range(start,end,csv_folder):
    df_dict = construct_rain_dicts(csv_folder)

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


if __name__ == '__main__':

    start = datetime.date(2004,1,1)
    end = datetime.date(2005,1,1)
    print(get_rain_4_date_range(start,end,'Rain Data'))
    #test = datetime.date(2000,11,1)
    #print(get_rain_4_date(test,'Rain Data'))
    #print(np.isnan(get_rain_4_date(test,'Rain Data')))

    #rain_dict = construct_rain_dicts('Rain Data')
    #pickle.dump(rain_dict, open("rain_dict.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


    



