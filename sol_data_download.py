from ceda_download import download_file
import os
from tqdm import tqdm


def download_sol_year(year, save_folder):
    url = f"https://dap.ceda.ac.uk/badc/ukmo-midas-open/data/uk-radiation-obs/dataset-version-202007/west-yorkshire/00534_bramham/qc-version-1/midas-open_uk-radiation-obs_dv-202007_west-yorkshire_00534_bramham_qcv-1_{year}.csv"
    download_file(url,save_folder)

def download_sol_data(save_folder):
    os.makedirs(save_folder)
    year_range = range(2001,2020)

    for year in tqdm(year_range):
        download_sol_year(year,save_folder)

def download_sol_yearly_full(year, save_folder):
    url = f"https://dap.ceda.ac.uk/badc/ukmo-midas/data/RO/yearly_files/midas_radtob_{year}01-{year}12.txt"
    download_file(url,save_folder)

def download_sol_data_full(year_range,save_folder):
    os.makedirs(save_folder)
    for year in tqdm(year_range):
        download_sol_yearly_full(year,save_folder)


if __name__ == '__main__':
    #download_sol_data('Solar Data')
    year_range = range(2001,2022)
    download_sol_data_full(year_range,'Full Solar Data')