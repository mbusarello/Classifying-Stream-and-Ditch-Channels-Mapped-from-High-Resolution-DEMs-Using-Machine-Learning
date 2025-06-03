"""
@author: Mariana Busarello, 2025
"""

import os
import argparse

try:
    import whitebox
    whitebox.download_wbt(linux_musl=True, reset=True)
    wbt = whitebox.WhiteboxTools()
except:
    #pass
    from WBT.whitebox_tools import WhiteboxTools
    whitebox.download_wbt(linux_musl=True, reset=True)
    wbt = WhiteboxTools()


def breaching_least_cost(input_path,output_path):
    for raster in sorted(os.listdir(input_path)):  
        if raster.endswith('.tif'):
            wbt.breach_depressions_least_cost(
                                    dem = input_path + raster, 
                                    output = output_path + raster,
                                    dist=100,
                                    max_cost=None, 
                                    min_dist=None, 
                                    flat_increment=None,
                                    fill=True
                                )
            

if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='breaches the DEMs',
                                     add_help=True,formatter_class=argparse.HelpFormatter)
    parser.add_argument('input_path', help='path to DEM tiles')
    parser.add_argument('output_path', help='destination path to the breached DEMs')
    args = vars(parser.parse_args())
    breaching_least_cost(**args)