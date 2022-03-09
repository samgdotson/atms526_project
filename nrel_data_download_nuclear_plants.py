
import numpy as np
import sys, os
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from nrel_api import *


coords = pd.read_csv('nuclear_plant_locations.csv', usecols=['Plant_Name', 'Longitude','Latitude'])

plant_names = list(coords.Plant_Name)
lons = list(coords.Longitude)
lats = list(coords.Latitude)

parameters['attr_list'] = ['air_temperature']

last_idx = np.where(np.array(plant_names)=='Beaver Valley')[0][0]


years = np.arange(2007,2021,1).astype('int')
full_frames = []
k = 0
N = len(plant_names[last_idx:])
for n, i, j in zip(plant_names[last_idx:], lats[last_idx:],lons[last_idx:]):
    print(f" ({k}/{N*len(years)}) Getting data for coordinates: {i}, {j} -- {n}")
    parameters['lon'] = j
    parameters['lat'] = i
    frames = []
    # get data for several years from this location
    if (k + len(years)) > 499:
        print('Download limit reached.')
        break
    for m,year in enumerate(years):
        print(f" ({k}/{N*len(years)}) -- {year}")
        parameters['year'] = year
        URL = make_csv_url(parameters=parameters, personal_data=personal_data, kind='solar')
#         print(URL)
        df = pd.read_csv(URL, skiprows=2)
        cols=['Year','Month', 'Day', 'Hour', 'Minute']
        df['time'] = pd.to_datetime(df[cols])
        df.drop(columns=cols, inplace=True)
        df.set_index('time', inplace=True)
        df.rename(columns={'Temperature':f'Temp_{n.replace(" ", "")}'}, inplace=True)
        frames.append(df)
        k += 1
    full_df = pd.concat(frames, axis=0)
    full_df.to_csv(f'nrel_psm_data/{n.replace(" ","")}_Temperature_2007_2020.csv')
    # full_frames.append(full_df)
    k += 1
# total_t = pd.concat(full_frames, axis=1)
