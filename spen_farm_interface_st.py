import streamlit as st

"""
imports required libraries
"""
import ee
import eemont
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import ipyleaflet
import ipywidgets as widgets

from scipy.signal import savgol_filter
import matplotlib.dates as mdates

import datetime

import geemap
from most_recent_product import field_closest_product_2_date_pkl
from read_temp_data import get_temps_4_date_pkl,get_temps_4_date_range_pkl
from read_rain_data import get_rain_4_date_pkl,get_rain_4_date_range_pkl
from read_sol_data import get_sol_4_date_pkl,get_sol_4_date_range_pkl
from read_cosmos_data import get_cosmos_col_4_date_pkl,get_cosmos_col_4_date_range_pkl

"""
defines some dictionaries and parameters for later use
"""
crop_lookup = {'OSR': 'Rapeseed Oil',
               'WW': 'Wheat',
               'WW2': 'Second Wheat',
               'WB': 'Winter Barley',
               'SB': 'Spring Barley',
               'PP': 'Permanent Pasture',
               'MZ': 'Maize',
               'VPEAS': 'Vining Peas',
               'WLIN': 'Winter Linseed',
               'FAL': 'Fallow',
               'POTS': 'Potatoes',
               'PIGS': 'Pigs',
               'LEY': 'Grass Ley',
              }

palette = {    
    'AGRO': 'ff0000',
    'IN HAND': '0000FF', 
    'NIAB': '00ff00', 
    'OSR': 'FFFF00', 
    'PIGS': 'ffb6c1', 
    'POTS': 'FFA500', 
    'PP': '006400', 
    'VPEAS': '800080',
    'WB': 'd2b48c', 
    'WW': 'F5DEB3',
    'WW2': '93856b',
    'LEY': 'FFFF99',
    'WLIN': 'b5651d',
    'FAL': 'C19A6B',
    '': 'FFFFFF',
    'UNKNOWN': 'FFFFFF'
}

vis_params = {
    'colorOpacity': .5,
    'width': 2,
    'lineType': 'dotted', 
    'fillColorOpacity': 0.65   
}

"""
creates the spen_farm class
"""
class spen_farm():
    def __init__(self):
        """
        defines map and downloads features from Earth Engine
        """
        self.Map = geemap.Map()
        self.aoi = ee.FeatureCollection("users/spiruel/aoi")
        self.aoi_area_s2 = 13480
        self.aoi_area_planet = 7907.351973668423
        self.planet_collection = ee.ImageCollection("users/spiruel/spen_planet_2021")
        self.fields = ee.FeatureCollection("users/zy18811/updated_fields")

        """
        list of field names
        """
        self.fields_list = self.fields.toList(100)
        self.field_names = [i['properties']['True_Field_name'] for i in self.fields_list.getInfo()]

        self.init_date = '2021-04-20' # date selected when app starts
        self.date = datetime.datetime.strptime(self.init_date,'%Y-%m-%d')
        self.sel_date_year = self.date.year

        """
        variables for use later
        """
        self.download_control = None
        self.selected_fc_geom = None
        self.flower_date = None
        
        self.min_date = datetime.datetime(2017,4,19)
        #unix2datetime = lambda x: datetime.datetime.fromtimestamp(x/1E3)
        #today = datetime.datetime.today().strftime('%Y-%m-%d')
        #self.min_planet_date = unix2datetime( self.planet_collection.first().get('system:time_start').getInfo() )
        self.min_planet_date = datetime.datetime(2020, 1, 1, 0, 0)
        #self.max_planet_date = unix2datetime( self.planet_collection.closest(today).first().get('system:time_start').getInfo() )

        """
        functions for map widgets
        """
        self.init_layers()
        self.field_dropdown()
        self.date_pick()
        self.desc_box()
        self.plot_button()
        
        self.ndvi_graph_exist = False
        self.temp_graph_exist = False
        self.prcp_graph_exist = False
        self.sol_graph_exist = False
        self.vwc_graph_exist = False
        
        """
        functions for plot widgets
        """
        self.ndvi_output = widgets.Output()
        self.ndvi_download = widgets.HTML()
    
        self.ndvi_tab = widgets.VBox([self.ndvi_output,self.ndvi_download])
        
        self.temp_output = widgets.Output()
        self.temp_download = widgets.HTML()
        
        self.temp_tab = widgets.VBox([self.temp_output,self.temp_download])
        
        self.prcp_output = widgets.Output()
        self.prcp_download = widgets.HTML()
        
        self.prcp_tab = widgets.VBox([self.prcp_output, self.prcp_download])

        self.sol_output = widgets.Output()
        self.sol_download = widgets.HTML()

        self.sol_tab = widgets.VBox([self.sol_output,self.sol_download])

        self.vwc_output = widgets.Output()
        self.vwc_download = widgets.HTML()

        self.vwc_tab = widgets.VBox([self.vwc_output,self.vwc_download])

        self.cosmos_csv = "COSMOS-UK_SPENF_HydroSoil_Daily_2013-2019.csv"
        self.crc_control()
        self.Map.on_interaction(self.handle_interaction)

    """
    defines initial layers and adds them to map
    """
    def init_layers(self):
        self.Map.setOptions('SATELLITE')

        # Sentinel 2 satellite imagery
        s2_col = ee.ImageCollection('COPERNICUS/S2_SR').filterBounds(self.aoi).closest(self.init_date).maskClouds().index(['NDVI'])
        self.s2 = s2_col.median().clip(self.aoi)

        # Sentinel 2 Cloud filtering
        self.masked_perc_s2 = (self.aoi_area_s2 - self.s2.clip(self.aoi).reduceRegion(
            'count', self.aoi, 20).get('CLOUD_MASK').getInfo())/self.aoi_area_s2

        # Field crop colours
        palette_crops = palette.copy()
        current_crops = [i['properties'][f"Y{self.date.year}"] for i in self.fields_list.getInfo()]
        for crop_pal in (set(palette_crops.keys()) - set(current_crops)):
            palette_crops.pop(crop_pal, None)
            
        self.Map.add_styled_vector(self.fields, column=f"Y{self.date.year}", palette=palette_crops, layer_name="fields", **vis_params)

        # Planet Satellite Imagery
        self.planet = self.planet_collection.closest(self.init_date).first()
        self.planet_ndvi = self.planet.normalizedDifference(['b4', 'b3']).rename('NDVI')
        if self.date >= self.min_planet_date:
            self.masked_perc_planet = (self.aoi_area_planet - self.planet.clip(self.aoi).mask().clip(self.aoi).reduceRegion(
                'sum', self.aoi, 20).get('b1').getInfo())/self.aoi_area_planet
        
            self.Map.addLayer(self.planet, {'bands':['b3','b2','b1'],'min':0, 'max':2000}, 'planet')
            self.Map.addLayer(self.planet_ndvi, {min: -1, max: 1, 'palette': ['blue', 'white', 'green']}, 'planet_ndvi')

        self.Map.addLayer(self.s2, {'bands':['B4','B3','B2'], 'min':0, 'max':2000}, 's2')
        self.Map.addLayer(self.s2.normalizedDifference(['B8', 'B4']).rename('NDVI'), {min: -1, max: 1, 'palette': ['blue', 'white', 'green']}, 's2_ndvi')

        self.Map.centerObject(self.fields, 14)

    """
    Widget for dropdown field selector
    """
    def field_dropdown(self):
        self.dropdown = widgets.Dropdown(
            options=sorted(self.field_names),
            value='Hanger Field',
            description='Field:'
        )

        self.dropdown.observe(self.handle_interaction, names='value')
            
        dropdown_output_control = ipyleaflet.WidgetControl(widget=self.dropdown, position='bottomleft')
        self.Map.add_control(dropdown_output_control)

    """
    Widget for field description box
    """
    def desc_box(self):
        self.output_widget = widgets.Output(layout={'border': '1px solid black'})
        output_control = ipyleaflet.WidgetControl(widget=self.output_widget, position='bottomright')
        self.desc_box_show = False

        """
        Widget button to show/hide description box
        """
        def handle_desc_box_toggle(args):
            if self.desc_box_show:
                self.Map.remove_control(output_control)
                self.hide_desc_box_toggle.description = 'Show Field Info.'
                self.hide_desc_box_toggle.tooltip = 'Shows field information box'
                self.hide_desc_box_toggle.icon = 'check'
                self.desc_box_show = False
            else:
                self.Map.add_control(output_control)
                self.hide_desc_box_toggle.description = 'Hide Field Info.'
                self.hide_desc_box_toggle.tooltip = 'Hides field information box'
                self.hide_desc_box_toggle.icon = 'times'
                self.desc_box_show = True


        self.hide_desc_box_toggle = widgets.Button(
            value=False,
            description='Show Field Info.',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Hides/shows field information box',
            icon='check' # (FontAwesome names without the `fa-` prefix)
        )

        self.hide_desc_box_toggle.on_click(handle_desc_box_toggle)

        desc_box_control = ipyleaflet.WidgetControl(widget=self.hide_desc_box_toggle, position='bottomright')
        self.Map.add_control(desc_box_control)

    """
    Function to handle clicking on field selection
    """
    def handle_interaction(self, *args, **kwargs):
        if len(args) > 0:
            self.selected_fc = self.fields.filterMetadata('True_Field_name', 'equals', args[0]['new'])
        elif kwargs.get('type') == 'click':
            latlon = kwargs.get('coordinates')
            self.Map.default_style = {'cursor': 'wait'}
            xy = ee.Geometry.Point(latlon[::-1])
            self.selected_fc = self.fields.filterBounds(xy)
        elif kwargs.get('type') != 'date_change':
            return

        with self.output_widget:
            self.output_widget.clear_output()
            try:
                self.selected_fc_name = self.selected_fc.first().get('True_Field_name').getInfo()
                self.selected_fc_num = self.selected_fc.first().get('True_Field_Number').getInfo()
                if len(self.selected_fc_num) == 3:
                    self.selected_fc_num += '0'

                date_str = self.date.strftime('%Y-%m-%d')
                prod_date_str = self.date.strftime('%d/%m/%Y')

                """
                Prints data to field description box and outlines selected field in red
                """
                print(f'{date_str}:')
                crop = self.selected_fc.first().get(f'Y{self.date.year}').getInfo()
                if crop in crop_lookup:
                    crop = crop_lookup[crop]

                self.selected_fc_geom = self.selected_fc.geometry()
                layer_desc = self.selected_fc_name + '\n' + crop
                self.Map.addLayer(ee.Image().paint(self.selected_fc_geom, 0, 2), {'palette': 'red'}, 'selected field')  
                print(layer_desc)

                if self.selected_fc_num == 'NULL':
                    print("No field inputs found for this field")
                else:
                    date_applied, product, rate_per_ha, units, n = field_closest_product_2_date_pkl(self.selected_fc_num,prod_date_str)

                    for i in range(n):
                        print(f"{rate_per_ha[i]} {units[i]} per ha of {product[i]} was applied on {date_applied}")

                degree_sign = u"\N{DEGREE SIGN}"
                soil, max_air, min_air = get_temps_4_date_pkl(self.date.date())
                if not np.isnan(soil):
                    print(f"Mean 10cm Soil Temp. = {soil:.2f}{degree_sign}C")
                else: 
                    print("No soil temperature data for this date")
                if not( np.isnan(max_air) and np.isnan(min_air)):
                    print(f"Mean Max Air Temp = {max_air:.2f}{degree_sign}C, Mean Min Air Temp = {min_air:.2f}{degree_sign}C")
                else:
                    print("No air temperature data for this date")

                rain_mm = get_rain_4_date_pkl(self.date.date())
                if not np.isnan(rain_mm):
                    print(f"Total precipitation = {rain_mm:.2f}mm")
                else:
                    print("No precipitation data for this date")

                sol_rad = get_sol_4_date_pkl(self.date.date())
                if not np.isnan(sol_rad):
                    print(f"Total solar irradiation = {sol_rad:.2f}KJ/m2")
                else:
                    print("No solar irradiation data for this date")


                albedo = get_cosmos_col_4_date_pkl('ALBEDO',self.date.date())
                if not np.isnan(albedo):
                    print(f"Albedo = {albedo}")
                else:
                    print("No albedo data for this date")

                vwc = get_cosmos_col_4_date_pkl('COSMOS_VWC',self.date.date())
                if not np.isnan(vwc):
                    print(f"CRNS VWC = {vwc}%")
                else:
                    print("No CRNS VWC data for this date")

                print(f'{100*self.masked_perc_s2:.0f}% S2 cloud filtered')
                if self.date >= self.min_planet_date:
                    print(f'{100*self.masked_perc_planet:.0f}% Planet cloud filtered')
                
                if self.dropdown.value != self.selected_fc_name:
                    self.dropdown.value = self.selected_fc_name
            except Exception as e:
                print('No feature could be found')
                try:
                    self.Map.remove_ee_layer('selected field')
                except Exception as e:
                    pass
        
        self.Map.default_style = {'cursor': 'pointer'}

    """
    Function to handle date selection dropdown
    """
    def handle_date_sel(self, *args):
        self.Map.default_style = {'cursor': 'wait'}
        date_str = args[0]['new'].strftime('%Y-%m-%d')
        self.date = datetime.datetime.combine(args[0]['new'], datetime.time())
        if  self.date < self.min_date or  self.date > datetime.datetime.today():
            self.Map.default_style = {'cursor': 'pointer'}
            return
        
        self.handle_crc({'new':False, 'old':None})
        self.toggle_button.value = False

        """
        Gets S2 and Planet images for selected date
        """
        s2_col = ee.ImageCollection('COPERNICUS/S2_SR').filterBounds(self.aoi).closest(date_str).maskClouds()
        self.s2 = s2_col.median().clip(self.aoi)
        self.masked_perc_s2 = (self.aoi_area_s2 - self.s2.clip(self.aoi).reduceRegion(
            'count', self.aoi, 20).get('CLOUD_MASK').getInfo())/self.aoi_area_s2
        
        self.planet = self.planet_collection.closest(date_str).first()
        self.planet_ndvi = self.planet.normalizedDifference(['b4', 'b3']).rename('NDVI')
        if self.date >= self.min_planet_date:
            self.masked_perc_planet = (self.aoi_area_planet - self.planet.clip(self.aoi).mask().clip(self.aoi).reduceRegion(
                'sum', self.aoi, 20).get('b1').getInfo())/self.aoi_area_planet

        palette_crops = palette.copy()
        current_crops = [i['properties'][f"Y{self.date.year}"] for i in self.fields_list.getInfo()]
        for crop_pal in (set(palette_crops.keys()) - set(current_crops)):
            palette_crops.pop(crop_pal, None)
            
        if self.date.year != self.sel_date_year:
            self.Map.add_styled_vector(self.fields, column=f"Y{self.date.year}", palette=palette_crops, layer_name="fields", **vis_params)
            self.sel_date_year = self.date.year
            
        if self.date >= self.min_planet_date:
            self.Map.addLayer(self.planet, {'bands':['b3','b2','b1'],'min':0, 'max':2000}, 'planet')
            self.Map.addLayer(self.planet_ndvi, {min: -1, max: 1, 'palette': ['blue', 'white', 'green']}, 'planet_ndvi')
        
        self.Map.addLayer(self.s2, {'bands':['B4','B3','B2'], 'min':0, 'max':2000}, 's2')
        self.Map.addLayer(self.s2.normalizedDifference(['B8', 'B4']).rename('NDVI'), {min: -1, max: 1, 'palette': ['blue', 'white', 'green']}, 's2_ndvi')

        if self.selected_fc_geom is not None:
            self.Map.addLayer(ee.Image().paint(self.selected_fc_geom, 0, 2), {'palette': 'red'}, 'selected field')
        
        self.handle_interaction(**{'type':'date_change'})
        self.Map.default_style = {'cursor': 'pointer'}

    """
    Widget for dropdown date selector
    """
    def date_pick(self):
        date_picker = widgets.DatePicker(
            description='Date',
            disabled=False,
            value=datetime.datetime.strptime(self.init_date,'%Y-%m-%d'),
            layout={'border': '1px solid black'}
        )

        date_picker.observe(self.handle_date_sel, names='value')

        date_output_control = ipyleaflet.WidgetControl(widget=date_picker, position='bottomleft')
        self.Map.add_control(date_output_control)


    """
    Widget for creating plots
    """
    def plot_button(self):
        plot_button = widgets.Button(
            description='Plot',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Plot data',
            icon='line-chart' # (FontAwesome names without the `fa-` prefix)
        )

        plot_button.on_click(self.handle_plot)

        self.button_output_control = ipyleaflet.WidgetControl(widget=plot_button, position='bottomright')
        self.Map.add_control(self.button_output_control)

    """
    Function for plotting COSMOS soil moisture
    """
    def handle_vwc_plot(self,*args):
        if self.date.date() > self.min_date.date():
            start =  max(self.min_date,self.date-datetime.timedelta(days=365)).date()
        else:
            start = (self.date-datetime.timedelta(days=365)).date()
        end = self.date.date()

        dd = [start + datetime.timedelta(days=x) for x in range((end - start).days + 1)]

        vwc_arr = get_cosmos_col_4_date_range_pkl('COSMOS_VWC',start,end)

        if np.isnan(vwc_arr).all():
            self.Map.default_style = {'cursor': 'pointer'}
            return -1

        fig = plt.figure(figsize = (6,3))

        plt.title("COSMOS Soil Moisture")
        plt.plot(dd, vwc_arr, label = 'CRNS VWC')

        plt.ylabel("VWC (%)")
        plt.grid(linestyle = 'dashed')
        plt.legend()

        ax = plt.gca()
        locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
        formatter = mdates.DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        with self.vwc_output:
            self.vwc_output.clear_output()
            plt.show()
            self.vwc_graph_exist = True

        df_download = pd.DataFrame()
        df_download['date'] = dd
        df_download['COSMOS VWC'] = vwc_arr
        df_download.to_csv('cosmos_vwc_data.csv', index=False)
        url = geemap.create_download_link('cosmos_vwc_data.csv').data.replace(
            'Click here to download:  download.csv','Download plot data')
        self.url = url

        self.vwc_download.value = url

        self.Map.default_style = {'cursor': 'pointer'}

    """
    Function for plotting solar irradiation and albedo
    """
    def handle_sol_plot(self, *args):
        if self.date.date() > self.min_date.date():
            start =  max(self.min_date,self.date-datetime.timedelta(days=365)).date()
        else:
            start = (self.date-datetime.timedelta(days=365)).date()
        end = self.date.date()

        dd = [start + datetime.timedelta(days=x) for x in range((end - start).days + 1)]

        sol_arr = get_sol_4_date_range_pkl(start,end)
        albedo_arr = get_cosmos_col_4_date_range_pkl('ALBEDO',start,end)

        if np.isnan(sol_arr).all() and np.isnan(albedo_arr).all():
            self.Map.default_style = {'cursor': 'pointer'}
            return -1

        fig,ax1 = plt.subplots(figsize = (6,3))

        color = 'tab:blue'
        plt.title("Solar Irradiation & Albedo")

        sol_line = ax1.plot(dd, sol_arr, label = 'Solar Irradiation',color=color)

        ax1.set_ylabel("Solar Irradiation (KJ/m2)")

        locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
        formatter = mdates.DateFormatter('%Y-%m-%d')
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(formatter)

        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel("Albedo")
        alb_line = ax2.plot(dd,albedo_arr,label = 'Albedo',color=color)
        
        locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
        formatter = mdates.DateFormatter('%Y-%m-%d')
        ax2.xaxis.set_major_locator(locator)
        ax2.xaxis.set_major_formatter(formatter)
        
        plt.grid(linestyle = 'dashed')
        
        lns = sol_line + alb_line
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc=0)

        fig.tight_layout()

        with self.sol_output:
            self.sol_output.clear_output()
            plt.show()
            self.sol_graph_exist = True

        df_download = pd.DataFrame()
        df_download['date'] = dd
        df_download['Solar Irradiation'] = sol_arr
        df_download['Albedo'] = albedo_arr
        df_download.to_csv('solar_irad_&_albedo_data.csv', index=False)
        url = geemap.create_download_link('solar_irad_&_albedo_data.csv').data.replace(
            'Click here to download:  download.csv','Download plot data')
        self.url = url

        self.sol_download.value = url

        self.Map.default_style = {'cursor': 'pointer'}

    """
    Function for plotting precipitation
    """
    def handle_prcp_plot(self,*args):
        self.Map.default_style = {'cursor': 'wait'}
        
        if self.date.date() > self.min_date.date():
            start =  max(self.min_date,self.date-datetime.timedelta(days=365)).date()
        else:
            start = (self.date-datetime.timedelta(days=365)).date()
        end = self.date.date()
        
        dd = [start + datetime.timedelta(days=x) for x in range((end - start).days + 1)]
        
        rain_arr = get_rain_4_date_range_pkl(start,end)
        
        if np.isnan(rain_arr).all():
            self.Map.default_style = {'cursor': 'pointer'}
            return -1
        
        fig = plt.figure(figsize = (6,3))
        
        plt.title("Precipitation")
        plt.plot(dd, rain_arr, label = 'Precipitation')
        
        plt.ylabel("Precipitation (mm)")
        plt.grid(linestyle = 'dashed')
        plt.legend()
        
        ax = plt.gca()
        locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
        formatter = mdates.DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        
        with self.prcp_output:
            self.prcp_output.clear_output()
            plt.show()
            self.prcp_graph_exist = True
            
        df_download = pd.DataFrame()
        df_download['date'] = dd
        df_download['Precipitation'] = rain_arr
        df_download.to_csv('precipitation_data.csv', index=False)
        url = geemap.create_download_link('precipitation_data.csv').data.replace(
            'Click here to download:  download.csv','Download plot data')
        self.url = url
        
        self.prcp_download.value = url
        
        self.Map.default_style = {'cursor': 'pointer'}

    """
    Function for plotting soil and air temperatures
    """
    def handle_temps_plot(self,*args):
        self.Map.default_style = {'cursor': 'wait'}

        if self.date.date() > self.min_date.date():
            start =  max(self.min_date,self.date-datetime.timedelta(days=365)).date()
        else:
            start = (self.date-datetime.timedelta(days=365)).date()
            
        end = self.date.date()
        
        dd = [start + datetime.timedelta(days=x) for x in range((end - start).days + 1)]
        
        temps_arr = get_temps_4_date_range_pkl(start,end)
        
        if np.isnan(temps_arr).all():
            self.Map.default_style = {'cursor': 'pointer'}
            return -1
        
        soil = temps_arr[:,0]
        air_max = temps_arr[:,1]
        air_min = temps_arr[:,2]
        
        fig = plt.figure(figsize=(6,3))
        
        plt.title("Temperature Data")
        plt.plot(dd,soil,label = '10 cm Soil Temp.')
        plt.plot(dd,air_max,label = 'Max. Air Temp.')
        plt.plot(dd,air_min,label = 'Min. Air Temp.')
        
        degree_sign = u"\N{DEGREE SIGN}"
        plt.ylabel(f'Temperature ({degree_sign}C)')
        plt.grid(linestyle='dashed')
        plt.legend()
        
        
        ax = plt.gca()
        locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
        formatter = mdates.DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        # Show the plot on the widget
        with self.temp_output:
            self.temp_output.clear_output()
            plt.show()
            self.temp_graph_exist = True
        
        
        df_download = pd.DataFrame()
        df_download['date'] = dd
        df_download['10 cm soil'] = soil
        df_download['max air'] = air_max
        df_download['min air'] = air_min
        df_download.to_csv('temperature_data.csv', index=False)
        url = geemap.create_download_link('temperature_data.csv').data.replace(
            'Click here to download:  download.csv','Download plot data')
        self.url = url
        
        self.temp_download.value = url

        self.Map.default_style = {'cursor': 'pointer'}

    """
    Function for plotting NDVI
    """
    def handle_ndvi_plot(self, *args):
        self.Map.default_style = {'cursor': 'wait'}
        
        output_widget = widgets.Output(layout={'border': '1px solid black'})
        output_control = ipyleaflet.WidgetControl(widget=output_widget, position='bottomright')
        self.Map.add_control(output_control)

        with output_widget:
            output_widget.clear_output()
            print('Plotting NDVI...')
            try:
                start_date_dt =  max(self.min_date,self.date-datetime.timedelta(days=365))
                start_date = start_date_dt.strftime('%Y-%m-%d')
                end_date_dt = self.date
                end_date = self.date.strftime('%Y-%m-%d')
                s2 = (ee.ImageCollection('COPERNICUS/S2_SR').filterDate(start_date,end_date)
                  .filterBounds(self.aoi)
                  .maskClouds()
                  .scale()
                  .index(['NDVI']))
                
                planet = self.planet_collection.filterDate(start_date,end_date).first()
                planet_ndvi = planet.normalizedDifference(['b4', 'b3']).rename('NDVI')
        
                ts = s2.getTimeSeriesByRegion(geometry = self.selected_fc_geom,
                                              bands = ['NDVI'],
                                              reducer = [ee.Reducer.mean(),ee.Reducer.stdDev()],
                                              scale = 20)

                print('Downloading data...')
                df = geemap.ee_to_pandas(ts)
                print('Downloaded data!')

                df.loc[df['NDVI'] < -999] = np.nan

                df.date = pd.to_datetime(df.date)
                df.date = df.date.apply(lambda x: x.replace(minute=0, second=0))

                df.loc[df.reducer=='mean'] = df.loc[df.reducer=='mean'].drop_duplicates('date')
                df.loc[df.reducer=='stdDev'] = df.loc[df.reducer=='stdDev'].drop_duplicates('date')

                df_std = df.loc[df.reducer=='stdDev']
                df_mean = df.loc[df.reducer=='mean']

                mean = df_mean.loc[df_mean.date.isin(df_std.date)].NDVI.values
                std = df_std.NDVI.values
                dates = df_mean.loc[df_mean.date.isin(df_std.date)].date
                                

                def addNDYI(img):
                    nd = img.normalizedDifference(['B3', 'B2']);
                    return img.addBands(nd.float().rename('NDYI'));

                crop = self.selected_fc.first().get(f'Y{self.date.year}').getInfo()
                if crop == 'OSR':
                    april1 = datetime.datetime(self.date.year,4,1)
                    if dates.min() <= april1 and dates.max() >= april1:
                        if any(dates.apply(lambda x: x.month).isin([3,4,5])):
                            #flowering_date = self.get_flowering()
                            print('Calculating flowering...')
                            s2 = s2.map(addNDYI)
                            ts_ndyi = s2.getTimeSeriesByRegion(geometry = self.selected_fc_geom,
                                                  bands = ['NDYI'],
                                                  reducer = [ee.Reducer.mean()],
                                                  scale = 40)

                            df_ndyi = geemap.ee_to_pandas(ts_ndyi)
                            df_ndyi.date = pd.to_datetime(df_ndyi.date)
                            
                            months = df_ndyi.date.apply(lambda x: x.month)
                            mask = (months >= 2) & (months <= 6)
                            df_ndyi = df_ndyi.loc[mask]
                            if len(df_ndyi) > 0:
                                df_ndyi.loc[df_ndyi['NDYI'] < -999] = np.nan
                                df_ndyi.date = df_ndyi.date.apply(lambda x: x.replace(minute=0, second=0))

                                df_ndyi.loc[df_ndyi.reducer=='mean'] = df_ndyi.loc[df_ndyi.reducer=='mean'].drop_duplicates('date')
                                df_mean_ndyi = df_ndyi.loc[df_ndyi.reducer=='mean']

                                self.flower_date = df_mean_ndyi.date.values[np.where(df_mean_ndyi.NDYI == df_mean_ndyi.NDYI.max())[0]][0]
                    else:
                        self.flower_date = None
                else:
                    self.flower_date = None

                def estimate_gaussian(dataset):

                    mu = np.mean(dataset)
                    sigma = np.std(dataset)
                    limit = sigma * 2

                    min_threshold = mu - limit
                    max_threshold = mu + limit

                    return mu, sigma, min_threshold, max_threshold

                mu, sigma, min_threshold, max_threshold = estimate_gaussian(mean)

                condition1 = (mean < min_threshold)
                condition2 = (mean > max_threshold)

                mask = ~(condition1 | condition2)
                dates = np.array(dates)[mask]
                mean = np.array(mean)[mask]
                std = np.array(std)[mask]


                fig = plt.figure(figsize=(6,3))

                plt.plot(dates, mean, '.-', label='mean')
                plt.fill_between(dates, mean-std, mean+std, alpha=.25)

                # apply SavGol filter
                if len(mean) > 9:
                    mean_savgol = savgol_filter(mean, window_length=9, polyorder=2)
                    plt.plot(dates, mean_savgol, label='savitsky-golay')

                if self.flower_date is not None:
                    plt.axvline(self.flower_date, alpha=.5, c='red', ls='--', label='est. flowering')

                plt.title(self.selected_fc_name)
                plt.ylim(0,1)
                plt.ylabel('NDVI')
                plt.grid(linestyle='dashed')
                plt.legend()

                ax = plt.gca()
                locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
                formatter = mdates.DateFormatter('%Y-%m-%d')
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)

                # Show the plot on the widget
                with self.ndvi_output:
                    self.ndvi_output.clear_output()
                    plt.show()
                    self.ndvi_graph_exist = True

                df_download = pd.DataFrame()
                df_download['date'] = dates
                df_download['mean'] = mean
                df_download['std'] = std
                df_download['mean_savgol'] = mean_savgol
                df_download.to_csv('ndvi.csv', index=False)
                url = geemap.create_download_link('ndvi.csv').data.replace(
                    'Click here to download:  download.csv','Download plot data')
                self.url = url
                self.ndvi_download.value = url
            

            except Exception as e:
                self.Map.default_style = {'cursor': 'pointer'}
                print('Error! Could not plot.')
                self.Map.remove_control(output_control)
                #raise Exception(e)
                return -1

            output_widget.clear_output()
            self.Map.remove_control(output_control)
            self.Map.default_style = {'cursor': 'pointer'}
    
    """
    Function handling all above plots
    """
    def handle_plot(self,*args):

        # removes plot button
        self.Map.remove_control(self.button_output_control)

        # creates plot selector widget
        data_select_widget = widgets.SelectMultiple(
                options=['NVDI','Temperature (Soil & Air)','Precipitation','Solar Irradiation & Albedo','COSMOS Soil Moisture','None'],
                #value=['NVDI'],
                #rows=10,
                description='Data to plot',
                disabled=False
                )
        data_select_control = ipyleaflet.WidgetControl(widget = data_select_widget, position='bottomright')
        self.Map.add_control(data_select_control)


        confirm_plot_widget = widgets.Button(
            description='Confirm Selection',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Plot Selection',
            icon='check' # (FontAwesome names without the `fa-` prefix)
            )

        """
        handles plots when sensors to plot has been confirmed
        """
        def handle_confirm_plot(*args):
            data_to_plot = data_select_widget.value
            self.Map.remove_control(data_select_control)
            self.Map.remove_control(confirm_plot_control)
            
            children = []
            titles = []
            for data_type in data_to_plot:
                if data_type == 'NVDI':
                    if self.handle_ndvi_plot(self,*args) == -1:
                        with self.ndvi_output:
                            self.ndvi_output.clear_output()
                            print("Error! Could not plot.")
                            print(f"Please check the date is after {self.min_date.date()} and that a field is selected.")
                            self.ndvi_graph_exist = True
                    children.append(self.ndvi_tab)
                    titles.append(data_type)
                elif data_type == 'Temperature (Soil & Air)':
                    if self.handle_temps_plot(self,*args) == -1:
                        with self.temp_output:
                            self.temp_output.clear_output()
                            print("Error! Could not plot.")
                            print("Please check there is temperature data available for this date in the field info box.")
                            self.temp_graph_exist = True
                    children.append(self.temp_tab)
                    titles.append(data_type)
                elif data_type == 'Precipitation':
                    if self.handle_prcp_plot(self, *args) == -1:
                        with self.prcp_output:
                            self.prcp_output.clear_output()
                            print("Error! Could not plot.")
                            print("Please check there is precipitation data available for this date in the field info box.")
                            self.prcp_graph_exist = True
                    children.append(self.prcp_tab)
                    titles.append(data_type)
                elif data_type == 'Solar Irradiation & Albedo':
                    if self.handle_sol_plot(self,*args) == -1:
                        with self.sol_output:
                            self.sol_output.clear_output()
                            print("Error! Could not plot.")
                            print("Please check there is solar irradiation/albedo data available for this date in the field info box.")
                            self.sol_graph_exist = True
                    children.append(self.sol_tab)
                    titles.append(data_type)
                elif data_type == 'COSMOS Soil Moisture':
                    if self.handle_vwc_plot(self,*args) == -1:
                        with self.vwc_output:
                            self.vwc_output.clear_output()
                            print("Error! Could not plot.")
                            print("Please check there is soil moisture (VWC) data available for this date in the field info box.")
                            self.vwc_graph_exist = True
                    children.append(self.vwc_tab)
                    titles.append(data_type)
                elif data_type == 'None':
                    self.Map.add_control(self.button_output_control)
                    return
            
            if len(children) != 0:
                tab = widgets.Tab()
                
                tab.children = children
                for i in range(len(children)):
                    tab.set_title(i,titles[i])
                tab_control = ipyleaflet.WidgetControl(widget = tab, position = 'bottomright')
                self.Map.add_control(tab_control)
            
            close_plot_widget = widgets.Button(
                description = 'Close plot(s)',
                disabled = False,
                button_style = '',
                tooltip = 'Close plot(s)',
                icon='times',
                )
            def handle_close_plot(*args):
                self.Map.remove_control(tab_control)
                self.Map.remove_control(close_plot_control)
                self.Map.add_control(self.button_output_control)

            close_plot_widget.on_click(handle_close_plot)
            close_plot_control = ipyleaflet.WidgetControl(widget = close_plot_widget, position = 'bottomright')
            
            any_graph_exist = self.ndvi_graph_exist or self.temp_graph_exist or self.prcp_graph_exist or self.sol_graph_exist or self.vwc_graph_exist
            
            if any_graph_exist:
                self.Map.add_control(close_plot_control)
            else:
                self.Map.add_control(self.button_output_control)
            
        confirm_plot_widget.on_click(handle_confirm_plot)
        confirm_plot_control = ipyleaflet.WidgetControl(widget = confirm_plot_widget, position = 'bottomright')
        self.Map.add_control(confirm_plot_control)
           
    """
    Function for displaying crop residue cover colour bar and map layer
    """
    def handle_crc(self, args):
        if type(args['old']) == dict:
            if 'value' in args['old'].keys():
                if args['old']['value']:
                    B2 = self.s2.select('B2')
                    B4 = self.s2.select('B4')
                    B12 = self.s2.select('B12')

                    crc = 100*(B2 - B4)/(B2 - B12)

                    vis_params_crc = {
                      'min': 0,
                      'max': 100,
                      'palette': ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725']}

                    self.Map.addLayer(crc, vis_params_crc, 'Crop Residue Cover')

                    if self.Map.colorbar is not None:
                        try:
                            self.Map.remove_colorbar()
                        except:
                            pass
                    self.Map.add_colorbar(vis_params_crc)
            
        elif not args['new']:
            if self.Map.colorbar is not None:
                try:
                    self.Map.remove_colorbar()
                    self.Map.remove_ee_layer('Crop Residue Cover')
                except:
                    pass

    """
    Widget for crop residue cover display
    """
    def crc_control(self):
        self.toggle_button = widgets.ToggleButton(
            value=False,
            description='Show CRC',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Crop Residue Cover',
            icon='pagelines' # (FontAwesome names without the `fa-` prefix)
        )

        self.toggle_button.observe(self.handle_crc)
        
        toggle_control = ipyleaflet.WidgetControl(widget=self.toggle_button, position='bottomright')
        self.Map.add_control(toggle_control)
        
"""
Renders map using spen_farm class
"""
interface = spen_farm()
m = interface.Map
#m.layout.width = '100%'
#m.layout.height = '800px'
m.to_streamlit()
