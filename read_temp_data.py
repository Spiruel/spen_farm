from temp_data_download import download_data
import numpy as np
import pandas as pd
import glob
import datetime
from tqdm import tqdm

def construct_temp_dicts(csv_folder_path):
    all_soil_csvs = glob.iglob(csv_folder_path + "/Soil Temp/*.csv")
    all_air_csvs = glob.iglob(csv_folder_path + "/Air Temp/*.csv")

    soil_df_dict = {}

    air_df_dict = {}

    for soil_csv, air_csv in zip(all_soil_csvs,all_air_csvs):
        soil_df = pd.read_csv(soil_csv,skiprows=85)
        air_df = pd.read_csv(air_csv,skiprows=90)

        year = soil_csv.split('.')[0][-4:]

        soil_df_dict[year] = soil_df
        air_df_dict[year] = air_df

    return soil_df_dict, air_df_dict


def daily_soil_temp(date,soil_df):
    daily_temp_10_cm_mean = soil_df.loc[pd.to_datetime(soil_df['ob_time']).dt.date == date]['q10cm_soil_temp'].mean()
    return daily_temp_10_cm_mean


def daily_air_temp(date,air_df):
    daily_temps_mean = air_df.loc[pd.to_datetime(air_df['ob_end_time']).dt.date == date][['max_air_temp','min_air_temp']].mean()
    max_temp_mean = daily_temps_mean['max_air_temp']
    min_temp_mean = daily_temps_mean['min_air_temp']
    return max_temp_mean,min_temp_mean


def get_temps_4_date(date,csv_folder):
    download_data(range(1959,2020),csv_folder)
    soil_dict, air_dict = construct_temp_dicts(csv_folder)
    try:
        soil_df = soil_dict[str(date.year)]
        soil_df = soil_df[:-1]
        soil_10_cm_mean = daily_soil_temp(date, soil_df)
    except KeyError:
        soil_10_cm_mean = np.NaN

    try:
        air_df = air_dict[str(date.year)]
        air_df = air_df[:-1]
        max_temp_mean, min_temp_mean = daily_air_temp(date, air_df)
    except KeyError:
        max_temp_mean = min_temp_mean = np.NaN

    return soil_10_cm_mean,max_temp_mean,min_temp_mean


def get_temps_4_date_range(start,end, csv_folder):
    download_data(range(1959,2020),csv_folder)
    soil_dict, air_dict = construct_temp_dicts(csv_folder)

    dd = [start + datetime.timedelta(days=x) for x in range((end - start).days + 1)]

    if start.year == end.year:
        try:
            soil_df = soil_dict[str(start.year)]
            soil_df = soil_df[:-1]
            daily_soil_means = soil_df.groupby(pd.to_datetime(soil_df['ob_time']).dt.date).mean()['q10cm_soil_temp']
            soil_arr = daily_soil_means[start:end].to_numpy()

        except KeyError:
            soil_arr = np.empty((end-start).days+1)
            soil_arr[:] = np.NaN

        try:
            air_df = air_dict[str(start.year)]
            air_df = air_df[:-1]
            daily_air_means = air_df.groupby(pd.to_datetime(air_df['ob_end_time']).dt.date).mean()[['max_air_temp','min_air_temp']]
            air_arr = daily_air_means[start:end].to_numpy()
        except KeyError:
            air_arr = np.empty(((end - start).days+1,2))
            air_arr[:] = np.NaN

        temps_arr = np.c_[soil_arr,air_arr]


    else:
        yy = range(start.year, end.year+1)
        temps_arr = np.empty(3)
        for y in yy:
            if y == start.year:
                y_start = start
                y_end = datetime.date(y,12,31)
            elif y == end.year:
                y_start = datetime.date(y,1,1)
                y_end = end
            else:
                y_start = datetime.date(y,1,1)
                y_end = datetime.date(y, 12, 31)

            n_days_between = (y_end - y_start).days+1
            try:
                soil_df = soil_dict[str(y)]
                soil_df = soil_df[:-1]
                daily_soil_means = soil_df.groupby(pd.to_datetime(soil_df['ob_time']).dt.date).mean()['q10cm_soil_temp']
                y_soil_arr = daily_soil_means[y_start:y_end].to_numpy()
            except KeyError:
                y_soil_arr = np.empty(n_days_between)
                y_soil_arr[:] = np.NaN

            try:
                air_df = air_dict[str(y)]
                air_df = air_df[:-1]
                daily_air_means = air_df.groupby(pd.to_datetime(air_df['ob_end_time']).dt.date).mean()[
                    ['max_air_temp', 'min_air_temp']]
                y_air_arr = daily_air_means[y_start:y_end].to_numpy()
            except KeyError:
                y_air_arr = np.empty((n_days_between,2))
                y_air_arr[:] = np.NaN
            y_temps_arr = np.c_[y_soil_arr, y_air_arr]
            temps_arr = np.vstack((temps_arr,y_temps_arr))
        temps_arr = temps_arr[1:,:]
    return temps_arr


if __name__ == '__main__':
    test_date = datetime.date(2019,1,1)
    end = datetime.date(2019,2,21)
    date1 = datetime.date(2017,2,26)

    start = end-datetime.timedelta(days=365)
    print(start,end)
    #print(get_temps_4_date(start, 'Temperature Data'))
    #print(get_temps_4_date(end_date, 'Temperature Data'))

    print(get_temps_4_date_range(start,end,'Temperature Data').shape)
    dd = np.array([start + datetime.timedelta(days=x) for x in range((end - start).days + 1)])
    print(dd.shape)


