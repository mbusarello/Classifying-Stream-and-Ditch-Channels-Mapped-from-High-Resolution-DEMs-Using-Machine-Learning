# -*- coding: utf-8 -*-
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
    
def subtracting(DEM_path,inference_path,output_path):
    for raster in os.listdir(inference_path):
        if raster.endswith('.tif'):
            wbt.subtract(
                    input1=os.path.join(DEM_path,raster),
                    input2=os.path.join(inference_path,raster),
                    output=os.path.join(output_path,raster)
                    )
            
if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='subtracts the combined probability maps from the DEM tiles',
                                     add_help=True,formatter_class=argparse.HelpFormatter)
    parser.add_argument('DEM_path', help='path to the location of the filled DEM tiles')
    parser.add_argument('probability_maps_path', help='input path to the location of the probability maps')
    parser.add_argument('output_path', help='destination path to the data after the subtraction')
    args = vars(parser.parse_args())
    subtracting(**args)