# -*- coding: utf-8 -*-
"""
@author: Mariana Busarello, 2024
"""

import os
import shutil
import geopandas as gpd
import pandas as pd
import argparse
try:
    import whitebox
    whitebox.download_wbt(linux_musl=True, reset=True)
    wbt = whitebox.WhiteboxTools()
except:
    from WBT.whitebox_tools import WhiteboxTools
    whitebox.download_wbt(linux_musl=True, reset=True)
    
def vectorizing_prediction(input_path,pointer_path,output_path):#
    for raster in os.listdir(input_path):
        if raster.endswith('.tif'):
            mosaic = raster.replace('.tif','.shp')
            in_mosaic = os.path.join(input_path,raster)
            pointer_mosaic = os.path.join(pointer_path,raster)
            out_mosaic = os.path.join(output_path,mosaic)


            wbt.raster_streams_to_vector(
                                    streams=in_mosaic,
                                    d8_pntr= pointer_mosaic,
                                    output=out_mosaic)
    
if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description="converts the breached U-Net channels from raster to vector, combining the data with the pointer rasters from the flow accumulation process",
                                     add_help=True,formatter_class=argparse.HelpFormatter)
    parser.add_argument('input_path',help='path to the DEMs breached during the burning process of the predicted channels from the U-Net')
    parser.add_argument('pointer_path', help='path to the pointer rasters output from flow accumulation')
    parser.add_argument('output_path', help='destination path to the vectorized stream lines')
    args = vars(parser.parse_args())
    vectorizing_prediction(**args)