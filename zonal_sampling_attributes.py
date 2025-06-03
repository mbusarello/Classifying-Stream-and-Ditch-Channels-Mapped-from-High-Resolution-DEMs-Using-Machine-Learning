"""
@author: Mariana Busarello, 2025
"""

import os
from rasterstats import zonal_stats
import geopandas as gpd
import argparse

def zonal_statistics(shp_path,raster_path,output_path,raster_type):    
    statistics = ["min","max","mean","median","std"]
    
    for file in os.listdir(raster_path):
        if file.endswith('.tif'):
            print(file)
            raster = os.path.join(raster_path,file)
            shapefile = os.path.join(shp_path,file.replace('.tif','.shp'))
            shp_read = gpd.read_file(shapefile)
            stats = zonal_stats(shapefile,raster,
                                        stats=statistics)
                    
            key_values = {key: [] for key in stats[0].keys()}
            for item in stats:
                        for key,value in item.items():
                            key_values[key].append(value)
                    
            for s in key_values:
                        shp_read[raster_type+'_'+s] = key_values[s]
                    
            shp_read.to_file(os.path.join(output_path,file.replace('.tif','.shp')))

if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='uses the buffer to sample the data rasters, giving the zonal statistics',
                                     add_help=True,formatter_class=argparse.HelpFormatter)
    parser.add_argument('shp_path', help='path to the buffered polylines with length')
    parser.add_argument('raster_path',help='path to the raster data to sample from')
    parser.add_argument('output_path',help='path to the output buffer with statistics calculated (adds new columns to the input one raster)')
    parser.add_argument('raster_type',help='type of topographic data raster being extracted (flow accumulation, average flowpath slope, etc)')
    args = vars(parser.parse_args())
    zonal_statistics(**args)