"""
For a given date, gets most recent product applied from Field Inputs data
"""
import pandas as pd
import datetime
import glob
import pickle


def closest_date(date_list, date):
    """
    returns the closest date to date in date_list
    :param date_list:
    :param date:
    :return:
    """
    return min(date_list, key=lambda x: abs(x - date))


def construct_field_dicts(csv_folder_path):
    """
    reads in field input data csvs and turns them into dicts of dataframes
    :param csv_folder_path: file path to field input csvs
    :return:
    """
    path = csv_folder_path
    all_csvs = glob.iglob(path+ "/*.csv")

    field_df_dict = {}
    field_id_dict = {}
    for csv in all_csvs:
        df = pd.read_csv(csv)
        FieldDefinedName = df["Field.Defined.Name"][0]
        try:
            field_name, field_id = FieldDefinedName.split('.')
        except ValueError:
            if FieldDefinedName.replace(' ','').isalpha():
                field_name = FieldDefinedName
                field_id = field_name
            else:
                field_name_i,field_name_ii,field_id = FieldDefinedName.split('.')
                field_name = ' '.join([field_name_i,field_name_ii])

        field_id_dict[field_name] = field_id
        field_df_dict[field_id] = df
    return field_df_dict, field_id_dict


def field_closest_date(field_id,field_df_dict,date):
    """
    For a given field, returns the closest date recorded to date
    :param field_id: field id number
    :param field_df_dict: dict of field dataframes
    :param date: date to find closest to
    :return:
    """
    field_df = field_df_dict[field_id]
    field_dates = pd.to_datetime(field_df["Actual.Issued.Date"]).dt.date.tolist()
    closest = closest_date(field_dates,date)
    return closest


def field_closest_product_2_date(field_name_or_id,date,csv_path,name=True,pkl = False):
    """
    For a given field, returns the product(s) applied on the closest date to date
    :param field_name_or_id: field name or id number
    :param date: date to get closest product to
    :param csv_path: file path to field input csvs
    :param name: Bool: True if field_name_or_id is field name, False if field id number
    :param pkl: Bool: True if field input data comes from .pkl instead of csvs
    :return:
    """
    if not pkl:
        field_df_dict, field_id_dict = construct_field_dicts(csv_path)
    else:
        (field_df_dict, field_id_dict) = pickle.load(open('field_inputs.pkl','rb'))

    if name:
        field_id = field_id_dict[field_name_or_id]
    else:
        field_id = field_name_or_id

    date = datetime.datetime.strptime(date,"%d/%m/%Y").date()

    closest_date = field_closest_date(field_id,field_df_dict,date)

    field_df = field_df_dict[field_id]
    closest_date_row = field_df.loc[ pd.to_datetime(field_df["Actual.Issued.Date"]).dt.date == closest_date]

    product = closest_date_row["Product.Name"].values
    rate_per_ha = closest_date_row["Rate.per.Application.Area.ha"].values
    units = closest_date_row["Units"].values
    date_applied = closest_date.strftime("%d/%m/%Y")

    return date_applied, product, rate_per_ha, units, len(product)


def field_closest_product_2_date_pkl(field_id,date):
    """
    For a field id number, gets the closest product applied to date
    :param field_id: field id number
    :param date: date
    :return:
    """
    return field_closest_product_2_date(field_id,date,'',name = False,pkl = True)


if __name__ == '__main__':

    path = "Field_Inputs/csv fields"
    #(field_df_dict,field_id_dict) = construct_field_dicts(path)
    #pickle.dump((field_df_dict,field_id_dict), open("field_inputs.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    date_applied, product, rate_per_ha, units, len = field_closest_product_2_date("0412","10/5/2018",path,name=False,pkl=True)

    for i in range(len):
        print(f"{rate_per_ha[i]} {units[i]} per ha of {product[i]} was applied on {date_applied}")





























