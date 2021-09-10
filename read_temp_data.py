"""
Reads MIDAS soil and air temperature data and returns values for given date/date range
"""
import numpy as np
import pandas as pd
import glob
import datetime
import pickle
from tqdm import tqdm


def construct_temp_dicts(csv_folder_path):
    """
    Constructs dicts of yearly soil and air temperature dataframes from csvs
    :param csv_folder_path: path to csvs
    :return:
    """
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
    """
    Returns soil temperature on date
    :param date: date
    :param soil_df: soil temp dataframe for year date falls in
    :return:
    """
    daily_temp_10_cm_mean = soil_df.loc[pd.to_datetime(soil_df['ob_time']).dt.date == date]['q10cm_soil_temp'].values[0]
    return daily_temp_10_cm_mean


def daily_air_temp(date,air_df):
    """
    Returns air temperature on data
    :param date: data
    :param air_df: air temp dataframe for year date falls in
    :return:
    """
    daily_temps_mean = air_df.loc[pd.to_datetime(air_df['ob_end_time']).dt.date == date][['max_air_temp','min_air_temp']]
    max_temp_mean = daily_temps_mean['max_air_temp'].values[0]
    min_temp_mean = daily_temps_mean['min_air_temp'].values[0]
    return max_temp_mean,min_temp_mean


def get_temps_4_date(date,csv_folder,pkl = False):
    """
    returns soil and air temperatures for date
    :param date: date
    :param csv_folder: path to data csvs
    :param pkl: Bool, True -> get dicts from .pkl not csvs
    :return:
    """
    if not pkl:
        soil_dict, air_dict = construct_temp_dicts(csv_folder)
    else:
        (soil_dict, air_dict) = pickle.load(open("full_temperature_dict.pkl",'rb'))

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
    """
    returns soil and air temperatures for date range
    :param start: start of date range
    :param end: end of date range
    :param csv_folder: path to data csvs
    :param pkl: Bool, True -> get dicts from .pkl not csvs
    :return:
    """
    if not pkl:
        soil_dict, air_dict = construct_temp_dicts(csv_folder)
    else:
        (soil_dict,air_dict) = pickle.load(open("full_temperature_dict.pkl",'rb'))

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
    """
    Returns temp data for date, getting dfs from .pkl
    :param date: date
    :return:
    """
    return get_temps_4_date(date,'',pkl = True)


def get_temps_4_date_range_pkl(start,end):
    """
    Returns temp data for date range, getting dfs from .pkl
    :param start: start of date range
    :param end: end of date range
    :return:
    """
    return get_temps_4_date_range(start,end,'',pkl=True)


def read_soil_temp_full(soil_file):
    """
    Reads soil temp data into dataframe from full MIDAS .txt
    :param soil_file: path to file
    :return:
    """
    df = pd.read_csv(soil_file, header=0, low_memory = False, names=['id',
                                                'id_type',
                                                'ob_time',
                                                'met_domain_name',
                                                'version_num',
                                                'src_id',
                                                'rec_st_ind',
                                                'q5cm_soil_temp',
                                                'q10cm_soil_temp',
                                                'q20cm_soil_temp',
                                                'q30cm_soil_temp',
                                                'q50cm_soil_temp',
                                                'q100cm_soil_temp',
                                                'q5cm_soil_temp_q',
                                                'q10cm_soil_temp_q',
                                                'q20cm_soil_temp_q',
                                                'q30cm_soil_temp_q',
                                                'q50cm_soil_temp_q',
                                                'q100cm_soil_temp_q',
                                                'meto_stmp_time',
                                                'midas_stmp_etime',
                                                'q5cm_soil_temp_j',
                                                'q10cm_soil_temp_j',
                                                'q20cm_soil_temp_j',
                                                'q30cm_soil_temp_j',
                                                'q50cm_soil_temp_j',
                                                'q100cm_soil_temp_j'])
    df = df[df.version_num == 1]
    df = df[df.src_id == 534][['ob_time','q10cm_soil_temp']]
    df['q10cm_soil_temp'] = pd.to_numeric(df['q10cm_soil_temp'], errors='coerce')
    return df


def read_air_temp_full(air_file):
    """
    Reads air temp data from full MIDAS .txt
    :param air_file: path to file
    :return:
    """
    df = pd.read_csv(air_file, header=0, low_memory=False, names=['ob_end_time',
                                                 'id_type',
                                                 'id',
                                                 'ob_hour_count',
                                                 'version_num',
                                                 'met_domain_name',
                                                 'src_id',
                                                 'rec_st_ind',
                                                 'max_air_temp',
                                                 'min_air_temp',
                                                 'min_grss_temp',
                                                 'min_conc_temp_',
                                                 'max_air_temp_q',
                                                 'min_air_temp_q',
                                                 'min_grss_temp_q',
                                                 'min_conc_temp_q',
                                                 'meto_stmp_time',
                                                 'midas_stmp_etime',
                                                 'max_air_temp_j',
                                                 'min_air_temp_j',
                                                 'min_grss_temp_j',
                                                 'min_conc_temp_j', ])

    df = df[df.version_num == 1]
    df = df[df.src_id == 534][['ob_end_time','max_air_temp','min_air_temp']]
    df['max_air_temp'] = pd.to_numeric(df['max_air_temp'], errors='coerce')
    df['min_air_temp'] = pd.to_numeric(df['min_air_temp'], errors='coerce')
    return df


def construct_temp_dicts_full(full_temp_path):
    """
    Constructs dicts of soil and air temperature dataframes from full MIDAS .txts
    :param full_temp_path: path to .txts
    :return:
    """
    all_soil_files = glob.iglob(full_temp_path + "/Full Soil Temp/*.txt")
    all_air_files = glob.iglob(full_temp_path + "/Full Air Temp/*.txt")

    full_soil_df_dict = {}

    full_air_df_dict = {}

    for soil_file, air_file in tqdm(zip(all_soil_files, all_air_files),total = 63):
        year = soil_file.split('.')[0][-6:-2]

        soil_df = read_soil_temp_full(soil_file)

        soil_df['ob_time'] = pd.to_datetime(soil_df['ob_time'])
        soil_df = soil_df.resample('D', on='ob_time', ).mean()
        ix = pd.date_range(datetime.date(int(year), 1, 1), datetime.date(int(year), 12, 31), freq='D')
        soil_df = soil_df.reindex(ix)
        soil_df = soil_df.reset_index(level=0).rename(columns={'index': 'ob_time'})

        air_df = read_air_temp_full(air_file)

        air_df['ob_end_time'] = pd.to_datetime(air_df['ob_end_time'])
        air_df = air_df.resample('D', on='ob_end_time', ).mean()
        ix = pd.date_range(datetime.date(int(year), 1, 1), datetime.date(int(year), 12, 31), freq='D')
        air_df = air_df.reindex(ix)
        air_df = air_df.reset_index(level=0).rename(columns={'index': 'ob_end_time'})

        full_soil_df_dict[year] = soil_df
        full_air_df_dict[year] = air_df

    return full_soil_df_dict, full_air_df_dict


if __name__ == '__main__':
    soil_dict, air_dict = construct_temp_dicts_full('Full Temperature Data')
    pickle.dump((soil_dict, air_dict), open("full_temperature_dict.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


