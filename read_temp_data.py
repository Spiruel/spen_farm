from temp_data_download import download_data
import numpy as np
import pandas as pd
import glob
import datetime
import pickle


def construct_temp_dicts(csv_folder_path):
    all_soil_csvs = glob.iglob(csv_folder_path + "/Soil Temp/*.csv")
    all_air_csvs = glob.iglob(csv_folder_path + "/Air Temp/*.csv")

    soil_df_dict = {}

    air_df_dict = {}

    for soil_csv, air_csv in zip(all_soil_csvs,all_air_csvs):
        year = soil_csv.split('.')[0][-4:]

        soil_df = pd.read_csv(soil_csv,skiprows=85)
        soil_df = soil_df[:-1]

        soil_df['ob_time'] = pd.to_datetime(soil_df['ob_time'])
        soil_df = soil_df.resample('D', on='ob_time', ).mean()

        ix = pd.date_range(datetime.date(int(year), 1, 1), datetime.date(int(year), 12, 31), freq='D')
        soil_df = soil_df.reindex(ix)
        soil_df = soil_df.reset_index(level=0).rename(columns={'index': 'ob_time'})

        air_df = pd.read_csv(air_csv,skiprows=90)
        air_df = air_df[:-1]

        air_df['ob_end_time'] = pd.to_datetime(air_df['ob_end_time'])
        air_df = air_df.resample('D', on='ob_end_time', ).mean()

        ix = pd.date_range(datetime.date(int(year), 1, 1), datetime.date(int(year), 12, 31), freq='D')
        air_df = air_df.reindex(ix)
        air_df = air_df.reset_index(level=0).rename(columns={'index': 'ob_end_time'})

        soil_df_dict[year] = soil_df
        air_df_dict[year] = air_df

    return soil_df_dict, air_df_dict


def daily_soil_temp(date,soil_df):
    daily_temp_10_cm_mean = soil_df.loc[pd.to_datetime(soil_df['ob_time']).dt.date == date]['q10cm_soil_temp'].values[0]
    return daily_temp_10_cm_mean


def daily_air_temp(date,air_df):
    daily_temps_mean = air_df.loc[pd.to_datetime(air_df['ob_end_time']).dt.date == date][['max_air_temp','min_air_temp']]
    max_temp_mean = daily_temps_mean['max_air_temp'].values[0]
    min_temp_mean = daily_temps_mean['min_air_temp'].values[0]
    return max_temp_mean,min_temp_mean


def get_temps_4_date(date,csv_folder,pkl = False):
    if not pkl:
        download_data(range(1959,2020),csv_folder)
        soil_dict, air_dict = construct_temp_dicts(csv_folder)
    else:
        (soil_dict, air_dict) = pickle.load(open("temperature_dict.pkl",'rb'))

    try:
        soil_df = soil_dict[str(date.year)]
        soil_10_cm_mean = daily_soil_temp(date, soil_df)
    except KeyError:
        soil_10_cm_mean = np.NaN

    try:
        air_df = air_dict[str(date.year)]
        max_temp_mean, min_temp_mean = daily_air_temp(date, air_df)
    except KeyError:
        max_temp_mean = min_temp_mean = np.NaN
    return soil_10_cm_mean,max_temp_mean,min_temp_mean


def get_temps_4_date_range(start,end, csv_folder, pkl = False):
    if not pkl:
        download_data(range(1959,2020),csv_folder)
        soil_dict, air_dict = construct_temp_dicts(csv_folder)
    else:
        (soil_dict,air_dict) = pickle.load(open("temperature_dict.pkl",'rb'))

    dd = [start + datetime.timedelta(days=x) for x in range((end - start).days + 1)]

    if start.year == end.year:
        try:
            soil_df = soil_dict[str(start.year)]
            soil_arr = soil_df[pd.to_datetime(soil_df['ob_time']).dt.date.between(start, end)]['q10cm_soil_temp'].to_numpy()

        except KeyError:
            soil_arr = np.empty((end-start).days+1)
            soil_arr[:] = np.NaN

        try:
            air_df = air_dict[str(start.year)]
            air_arr = air_df[pd.to_datetime(air_df['ob_end_time']).dt.date.between(start,end)][['max_air_temp','min_air_temp']].to_numpy()

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
                y_soil_arr = soil_df[pd.to_datetime(soil_df['ob_time']).dt.date.between(y_start, y_end)]['q10cm_soil_temp'].to_numpy()
            except KeyError:
                y_soil_arr = np.empty(n_days_between)
                y_soil_arr[:] = np.NaN

            try:
                air_df = air_dict[str(y)]
                y_air_arr = air_df[pd.to_datetime(air_df['ob_end_time']).dt.date.between(y_start,y_end)][['max_air_temp','min_air_temp']].to_numpy()
            except KeyError:
                y_air_arr = np.empty((n_days_between,2))
                y_air_arr[:] = np.NaN
            y_temps_arr = np.c_[y_soil_arr, y_air_arr]
            temps_arr = np.vstack((temps_arr,y_temps_arr))
        temps_arr = temps_arr[1:,:]
    return temps_arr


def get_temps_4_date_pkl(date):
    return get_temps_4_date(date,'',pkl = True)


def get_temps_4_date_range_pkl(start,end):
    return get_temps_4_date_range(start,end,'',pkl=True)


if __name__ == '__main__':
    '''
    test_date = datetime.date(2012,3,20)
    end = datetime.date(2014,2,6)
    date1 = datetime.date(2012,2,26)

    start = (end - datetime.timedelta(days=365))
    print(start,end)
    #print(get_temps_4_date(start, 'Temperature Data'))
    #print(get_temps_4_date(end_date, 'Temperature Data'))

    print(get_temps_4_date_range(start,end,'Temperature Data').shape)
    dd = np.array([start + datetime.timedelta(days=x) for x in range((end - start).days + 1)])
    print(dd.shape)
    '''
    soil_dict, air_dict = construct_temp_dicts('Temperature Data')
    pickle.dump((soil_dict,air_dict),open("temperature_dict.pkl",'wb'),protocol=pickle.HIGHEST_PROTOCOL)
    #test_date = datetime.date(2019, 1, 1)
    #test_datee = datetime.date(2019, 1, 10)
    #print(get_temps_4_date(test_date,'Temperature Data',pkl = True))
    #print(get_temps_4_date_range(test_date,test_datee,'Temperature Data',pkl = True))


