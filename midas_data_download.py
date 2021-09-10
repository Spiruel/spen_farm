"""
NOT USED IN CURRENT VERSION OF spen_farm_interface.ipynb

Downloads MIDAS data from CEDA for given years
"""
from ceda_download import download_file
import os
from tqdm import tqdm

def download_soil_temp_full(year,save_folder):
    url = f"https://dap.ceda.ac.uk/badc/ukmo-midas/data/ST/yearly_files/midas_soiltemp_{year}01-{year}12.txt"
    download_file(url, save_folder)


def download_air_temp_full(year,save_folder):
    url = f"https://dap.ceda.ac.uk/badc/ukmo-midas/data/TD/yearly_files/midas_tempdrnl_{year}01-{year}12.txt"
    download_file(url,save_folder)


def download_temp_data_full(year_range,save_folder):
    soil_folder = os.path.join(save_folder, "Full Soil Temp")
    air_folder = os.path.join(save_folder, "Full Air Temp")

    os.makedirs(soil_folder)
    os.makedirs(air_folder)

    for year in tqdm(year_range):
        download_soil_temp_full(year,soil_folder)
        download_air_temp_full(year,air_folder)


def download_daily_rain_full(year, save_folder):
    url = f"https://dap.ceda.ac.uk/badc/ukmo-midas/data/RD/yearly_files/midas_raindrnl_{year}01-{year}12.txt"
    download_file(url, save_folder)


def download_hourly_rain_full(year, save_folder):
    url = f"https://dap.ceda.ac.uk/badc/ukmo-midas/data/RH/yearly_files/midas_rainhrly_{year}01-{year}12.txt"
    download_file(url, save_folder)


def download_rain_data_full(year_range, save_folder):
    daily_folder = os.path.join(save_folder, "Full Daily")
    hourly_folder = os.path.join(save_folder, "Full Hourly")

    os.makedirs(daily_folder)
    os.makedirs(hourly_folder)

    for year in tqdm(year_range):
        download_daily_rain_full(year,daily_folder)
        download_hourly_rain_full(year,hourly_folder)


def download_sol_yearly_full(year, save_folder):
    url = f"https://dap.ceda.ac.uk/badc/ukmo-midas/data/RO/yearly_files/midas_radtob_{year}01-{year}12.txt"
    download_file(url,save_folder)


def download_sol_data_full(year_range,save_folder):
    os.makedirs(save_folder)
    for year in tqdm(year_range):
        download_sol_yearly_full(year,save_folder)


def download_midas_data_full(year_range):
    download_temp_data_full(year_range, 'Full Temperature Data')
    download_rain_data_full(year_range, 'Full Rain Data')
    download_sol_data_full(year_range, 'Full Solar Data')