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

def combined_probability(input_path,output_path):
    for pr in os.listdir(input_path):
        if pr.endswith('1_prob.tif'):
            print(pr)
            input_raster1 = os.path.join(input_path,pr)
            input_raster2 = os.path.join(input_path,pr.replace('1_prob','2_prob'))
            output_raster = os.path.join(output_path,pr.replace('_class_1_prob',''))
            wbt.add(
                    input_raster1, 
                    input_raster2, 
                    output_raster
                    )

if __name__== '__main__':
    parser = argparse.ArgumentParser(
        description='combines the probability of ditches and streams into a single raster',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_path', help='path to the probability maps with the new threshold')     
    parser.add_argument('output_path', help='path to the combined probability maps')

    args = vars(parser.parse_args())
    combined_probability(**args)