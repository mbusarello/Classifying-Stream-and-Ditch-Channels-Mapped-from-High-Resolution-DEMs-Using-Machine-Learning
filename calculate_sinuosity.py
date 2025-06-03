"""
@author: Mariana Busarello, 2025
"""

import geopandas as gpd
import os
from shapely.geometry import LineString
import argparse

def sinuosity(line_path,buffer_path):
    for shapefile in os.listdir(line_path):
        sinuosity = []
        if shapefile.endswith('.shp'):
            print(shapefile)
            shp = gpd.read_file(os.path.join(line_path,shapefile))
            for idx, row in shp.iterrows():
                line = row['geometry']    
                start_point = line.coords[0]
                end_point = line.coords[-1]
                straight_line = LineString((start_point, end_point))
                try:
                    sinu = line.length / straight_line.length
                    sinuosity.append(sinu)
                except Exception as ZeroDivisionError:
                    sinuosity.append(0)
            
            shp['sinuosity'] = sinuosity
            shp.to_file(os.path.join(line_path,shapefile))
            try:
                bff = gpd.read_file(os.path.join(buffer_path,shapefile))
                bff['sinuosity'] = sinuosity
                bff.to_file(os.path.join(buffer_path,shapefile))
            except Exception as e:
                print(e)

if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='calculates the sinuosity from the vector lines and transfer it to the buffers',
                                     add_help=True,formatter_class=argparse.HelpFormatter)
    parser.add_argument('line_path', help='path to the polylines to be analyzed')
    parser.add_argument('buffer_path', help="path to the buffers to store the results")
    args = vars(parser.parse_args())
    sinuosity(**args)            
