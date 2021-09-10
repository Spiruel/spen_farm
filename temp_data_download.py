from ceda_download import download_file
import os
from tqdm import tqdm


def download_soil_temp(year,save_folder):
    url = f"https://dap.ceda.ac.uk/badc/ukmo-midas-open/data/uk-soil-temperature-obs/dataset-version-202007/west-yorkshire/00534_bramham/qc-version-1/midas-open_uk-soil-temperature-obs_dv-202007_west-yorkshire_00534_bramham_qcv-1_{year}.csv"
    download_file(url,save_folder)


def download_air_temp(year,save_folder):
    url = f"https://dap.ceda.ac.uk/badc/ukmo-midas-open/data/uk-daily-temperature-obs/dataset-version-202007/west-yorkshire/00534_bramham/qc-version-1/midas-open_uk-daily-temperature-obs_dv-202007_west-yorkshire_00534_bramham_qcv-1_{year}.csv"
    download_file(url,save_folder)


def download_temp_data(year_range,save_folder):
    download_soil = True
    download_air = True

    soil_folder = os.path.join(save_folder, "Soil Temp")
    air_folder = os.path.join(save_folder, "Air Temp")

    if not os.path.isdir(save_folder):
        os.makedirs(soil_folder)
        os.makedirs(air_folder)

    else:
        if not os.path.isdir(soil_folder):
            os.makedirs(soil_folder)
        else:
            download_soil = False

        if not os.path.isdir(air_folder):
            os.makedirs(air_folder)
        else:
            download_air = False

    if download_soil or download_air:
        for year in tqdm(year_range):
            if download_soil:
                download_soil_temp(year, soil_folder)
            if download_air:
                download_air_temp(year, air_folder)


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


if __name__ == '__main__':
    year_range = range(1959,2022)
    download_temp_data_full(year_range,'Full Temperature Data')



