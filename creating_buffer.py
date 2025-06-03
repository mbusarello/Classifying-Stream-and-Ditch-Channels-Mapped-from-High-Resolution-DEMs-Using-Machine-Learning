"""
@author: Mariana Busarello, 2025
"""

import os
import geopandas as gpd
import shapely
from shapely import LineString, MultiLineString
import argparse

def buffers(input_path,output_path):
    for shape in os.listdir(input_path):
        if shape.endswith('.shp'):
            print(shape)
            lines = []
            shp = gpd.read_file(os.path.join(input_path,shape))
            for line in shp.geometry:
                if isinstance(line, LineString):
                    small_line = shapely.buffer(line,distance=3)
                    lines.append(small_line)
                elif isinstance(line, MultiLineString):
                    for part in line:
                        small_line = shapely.buffer(line,distance=3)
                        lines.append(small_line)
        
            df = gpd.GeoDataFrame(geometry=lines,crs=shp.crs)
            df['length'] = shp['length']
            df.to_file(os.path.join(output_path+shape))

if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='creates buffers of 6m around the vector lines',
                                     add_help=True,formatter_class=argparse.HelpFormatter)
    parser.add_argument('input_path',help='input path to the split vector lines')
    parser.add_argument('output_path',help='destination path to the buffers')
    args = vars(parser.parse_args())
    buffers(**args)