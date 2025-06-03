"""
@author: Mariana Busarello, 2025
"""

import os
import rasterio as rio
import numpy as np
import argparse

try:
    import whitebox
    whitebox.download_wbt(linux_musl=True, reset=True)
    wbt = whitebox.WhiteboxTools()
except:
    from WBT.whitebox_tools import WhiteboxTools
    whitebox.download_wbt(linux_musl=True, reset=True)
    wbt = WhiteboxTools()
    

def average_flowpath_slope(input_path,output_path):
    for raster in sorted(os.listdir(input_path)):  
        if raster.endswith('.tif'):
            wbt.average_flowpath_slope(
                                    dem = input_path + raster, 
                                    output = output_path + raster
                                        )
            

if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='calculates the average flowpath slope',
                                     add_help=True,formatter_class=argparse.HelpFormatter)
    parser.add_argument('input_path', help='path to breached DEM mosaics')
    parser.add_argument('output_path', help='destination path')
    args = vars(parser.parse_args())
    average_flowpath_slope(**args) 