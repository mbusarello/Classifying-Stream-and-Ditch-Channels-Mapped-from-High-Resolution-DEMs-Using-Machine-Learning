"""
@author: Mariana Busarello, 2025
"""

import os
import pandas as pd
import numpy as np
import rasterio as rio
import argparse
import geopandas as gpd

try:
    import whitebox
    whitebox.download_wbt(linux_musl=True, reset=True)
    wbt = whitebox.WhiteboxTools()
except:
    from WBT.whitebox_tools import WhiteboxTools
    whitebox.download_wbt(linux_musl=True, reset=True)
    wbt = WhiteboxTools()

def mosaic(tiles_path,mosaic_path):
    zones = {'DemMosaicZone2.shp':['18H005_73025_8125_25.tif', '18H005_73025_8150_25.tif', '18H005_73025_8175_25.tif', '18H005_73025_8200_25.tif', '18H005_73025_8225_25.tif', '18H005_73050_8125_25.tif', '18H005_73050_8175_25.tif', '18H005_73050_8150_25.tif', '18H005_73050_8200_25.tif', '18H005_73050_8225_25.tif', '18H005_73075_8125_25.tif', '18H005_73075_8150_25.tif', '18H005_73075_8175_25.tif', '18H005_73075_8200_25.tif', '18H005_73075_8225_25.tif', '18H005_73100_8125_25.tif', '18H005_73100_8150_25.tif', '18H005_73100_8175_25.tif', '18H005_73100_8200_25.tif', '18H005_73100_8225_25.tif'],
     'DemMosaicZone5.shp':['18E023_68925_6250_25.tif', '18E023_68925_6275_25.tif', '18E023_68925_6300_25.tif', '18E023_68950_6275_25.tif', '18E023_68950_6250_25.tif', '18E023_68950_6300_25.tif', '18E023_68975_6300_25.tif', '18E023_68975_6275_25.tif', '18E023_68975_6250_25.tif'],
    'DemMosaicZone7.shp':['19B001_63575_3400_25.tif', '19B001_63575_3425_25.tif', '19B001_63575_3450_25.tif', '19B001_63575_3375_25.tif', '19B001_63600_3375_25.tif', '19B001_63600_3450_25.tif', '19B001_63600_3425_25.tif', '19B001_63600_3400_25.tif', '19B001_63625_3375_25.tif', '19B001_63625_3400_25.tif', '19B001_63625_3425_25.tif', '19B001_63625_3450_25.tif', '19B001_63650_3400_25.tif', '19B001_63650_3425_25.tif', '19B001_63650_3375_25.tif', '19B001_63650_3450_25.tif', '19B001_63675_3400_25.tif', '19B001_63675_3425_25.tif', '19B001_63675_3450_25.tif', '19B001_63700_3425_25.tif', '19B001_63700_3400_25.tif', '19B001_63700_3450_25.tif', '19B001_63725_3400_25.tif', '19B001_63725_3425_25.tif', '19B001_63725_3450_25.tif'],
    
    'DemMosaicZone9.shp':['19A017_62350_5000_25.tif', '19A017_62350_5050_25.tif', '19A017_62350_5025_25.tif', '19A017_62350_5075_25.tif', '19A017_62375_5000_25.tif', '19A017_62375_5025_25.tif', '19A017_62375_5050_25.tif', '19A017_62375_5075_25.tif', '19A017_62400_5000_25.tif', '19A017_62400_5025_25.tif', '19A017_62400_5050_25.tif', '19A017_62400_5075_25.tif']}
    
    for zone in zones:
        raster_list = [os.path.join(tiles_path, raster) for raster in zones[zone]]
        rasters = ';'.join(raster_list)
        wbt.mosaic(output=os.path.join(mosaic_path,zone),
                   inputs=rasters,
                   method='nn')

if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='mosaics the DEM and probability maps',
                                     add_help=True,formatter_class=argparse.HelpFormatter)
    parser.add_argument('tiles_path', help='path to the DEM or probability map tiles')
    parser.add_argument('mosaic_path',help='destination path to the mosaics')
    args = vars(parser.parse_args())
    mosaic(**args)