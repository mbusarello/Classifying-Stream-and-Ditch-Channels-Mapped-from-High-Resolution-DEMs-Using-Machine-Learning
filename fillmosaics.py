"""
@author: Mariana Busarello, 2024
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


def filldem(input_path,output_path):
    for dem in os.listdir(input_path):
        if dem.endswith('.tif'):
            wbt.fill_missing_data(
                i = input_path+dem, 
                output = output_path+dem, 
                filter=666, 
                weight=2.0, 
                no_edges=False
                                )

            

if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='fills the missing data from the raw DEM mosaics',
                                     add_help=True,formatter_class=argparse.HelpFormatter)
    parser.add_argument('input_path', help='path to the mosaics')
    parser.add_argument('output_path', help='destination path to the depressionless mosaics')
    args = vars(parser.parse_args())
    filldem(**args)    
