"""
@author: Mariana Busarello, 2025
"""

import geopandas as gpd
import os
import argparse

def length(input_path,output_path):
    for shapefile in os.listdir(input_path):
        if shapefile.endswith('.shp'):
            mosaic = os.path.join(input_path,shapefile)
            print(mosaic)
            shape = gpd.read_file(mosaic)
            shape['length'] = shape['geometry'].length
            shape.to_file(mosaic.replace(input_path,output_path))
            
if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='calculates the length of the vector lines after the split',
                                     add_help=True,formatter_class=argparse.HelpFormatter)
    parser.add_argument('input_path', help='path to the polyline vectors')
    parser.add_argument('output_path',help='path to the output vectors with the calculated length')
    args = vars(parser.parse_args())
    length(**args)