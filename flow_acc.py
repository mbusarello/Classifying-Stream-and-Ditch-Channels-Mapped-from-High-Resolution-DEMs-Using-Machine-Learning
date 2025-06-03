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
    

def facc(breached_dir, pointer_dir, accumulation_dir):
    for watershed in os.listdir(breached_dir):
        if watershed.endswith('.tif'):
            wbt.d8_pointer(
                dem = breached_dir + watershed, 
                output = pointer_dir + watershed, 
                esri_pntr=False
            )

            wbt.d8_flow_accumulation(
            i = pointer_dir + watershed, 
            output = accumulation_dir + watershed, 
            out_type='catchment area', 
            log=False, 
            clip=False, 
            pntr=True, 
            esri_pntr=False
            )


if __name__== '__main__':
    parser = argparse.ArgumentParser(
        description='Calculates the flow accumulation and flow direction from the breached DEMs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('breached_dir', help='preprocessed breached DEM')     
    parser.add_argument('pointer_dir', help='path to directory where the flow pointer grids will be stored')
    parser.add_argument('accumulation_dir', help='path to directory where the flow accumulation grids will be stored')

    args = vars(parser.parse_args())
    facc(**args)