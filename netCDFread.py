# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 12:04:21 2018

@author: Brian
"""

from netCDF4 import Dataset
from netCDF4 import MFDataset
dataset = Dataset('data2.nc')

d_lat = dataset.variables['lat'][:]
d_lon = dataset.variables['lon'][:]