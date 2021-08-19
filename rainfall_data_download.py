from ceda_download import download_file
import os
from tqdm import tqdm

def download_daily_rain(year, save_folder):
    url = f"https://dap.ceda.ac.uk/badc/ukmo-midas-open/data/uk-daily-rain-obs/dataset-version-202007/west-yorkshire/00534_bramham/qc-version-1/midas-open_uk-daily-rain-obs_dv-202007_west-yorkshire_00534_bramham_qcv-1_{year}.csv"
    download_file(url, save_folder)

def download_hourly_rain(year, save_folder):
    url = f"https://dap.ceda.ac.uk/badc/ukmo-midas-open/data/uk-hourly-rain-obs/dataset-version-202007/west-yorkshire/00534_bramham/qc-version-1/midas-open_uk-hourly-rain-obs_dv-202007_west-yorkshire_00534_bramham_qcv-1_{year}.csv"
    download_file(url, save_folder)

def download_rain(save_folder):

    daily_folder = os.path.join(save_folder, "Daily")
    hourly_folder = os.path.join(save_folder, "Hourly")

    os.makedirs(daily_folder)
    os.makedirs(hourly_folder)

    daily_year_range = range(1961,2001)
    hourly_year_range = range(2001,2020)

    for year in tqdm(daily_year_range):
        download_daily_rain(year,daily_folder)
    for year in tqdm(hourly_year_range):
        download_hourly_rain(year,hourly_folder)

if __name__ == '__main__':
    download_rain('Rain Data')