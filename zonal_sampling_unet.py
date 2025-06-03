"""
@author: Mariana Busarello, 2025
"""

import os
from rasterstats import zonal_stats
import geopandas as gpd
import argparse

def zonal_class(shp_path,raster_path,output_path):       
    for file in os.listdir(shp_path):
        if file.endswith('.shp'):
            print(file)
            raster = os.path.join(raster_path,file.replace('.shp','.tif'))
            shapefile = os.path.join(shp_path,file)
            shp_read = gpd.read_file(shapefile)
            classes = zonal_stats(shapefile,raster,categorical=True)
            majority_classes = []
            
            for idx, stat in enumerate(classes): 
                if isinstance(stat, dict): 
                    max_class = None
                    max_count = 0
                    for cls, count in stat.items():
                        column_name = f"class_{int(cls)}" 
                        if column_name not in shp_read.columns:
                            shp_read[column_name] = 0  
                        shp_read.at[idx, column_name] = count 
        
                        if count > max_count:
                            max_class = cls
                            max_count = count
        
                    
                    majority_classes.append(max_class)
                else:
                    majority_classes.append(None)

    
            shp_read["class_dl"] = majority_classes
            shp_read['class_dl'] = shp_read['class_dl'].replace({1: 2, 2: 1}) #the U-Net outputs the channels with the opposite labels, this line fixes that to match the original class from the polylines
            shp_read = shp_read.rename(columns={"class_1": "class_2", "class_2": "class_1"})           
            shp_read.to_file(os.path.join(output_path,file))

if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='uses the buffer to sample the deep learning inference output, calculating the majority channel type class for each buffer. it also returns the count of each class.',
                                     add_help=True,formatter_class=argparse.HelpFormatter)
    parser.add_argument('shp_path', help='path to the buffered polylines')
    parser.add_argument('raster_path',help='path to the deep learning inference to sample from')
    parser.add_argument('output_path',help='path to the output buffer with statistics calculated (adds new columns to the input one)')
    args = vars(parser.parse_args())
    zonal_class(**args)