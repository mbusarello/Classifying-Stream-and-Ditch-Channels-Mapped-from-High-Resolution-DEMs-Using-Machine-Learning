"""
@author: Mariana Busarello, 2025
"""

import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
import argparse

def class_sampling(buffer_path,output_path,gt_path):    
    zones = {'DemMosaicZone2.shp':['18H005_73025_8125_25', '18H005_73025_8150_25', '18H005_73025_8175_25', '18H005_73025_8200_25', '18H005_73025_8225_25', '18H005_73050_8125_25', '18H005_73050_8175_25', '18H005_73050_8150_25', '18H005_73050_8200_25', '18H005_73050_8225_25', '18H005_73075_8125_25', '18H005_73075_8150_25', '18H005_73075_8175_25', '18H005_73075_8200_25', '18H005_73075_8225_25', '18H005_73100_8125_25', '18H005_73100_8150_25', '18H005_73100_8175_25', '18H005_73100_8200_25', '18H005_73100_8225_25'],
     'DemMosaicZone5.shp':['18E023_68925_6250_25', '18E023_68925_6275_25', '18E023_68925_6300_25', '18E023_68950_6275_25', '18E023_68950_6250_25', '18E023_68950_6300_25', '18E023_68975_6300_25', '18E023_68975_6275_25', '18E023_68975_6250_25'],
    'DemMosaicZone7.shp':['19B001_63575_3400_25', '19B001_63575_3425_25', '19B001_63575_3450_25', '19B001_63575_3375_25', '19B001_63600_3375_25', '19B001_63600_3450_25', '19B001_63600_3425_25', '19B001_63600_3400_25', '19B001_63625_3375_25', '19B001_63625_3400_25', '19B001_63625_3425_25', '19B001_63625_3450_25', '19B001_63650_3400_25', '19B001_63650_3425_25', '19B001_63650_3375_25', '19B001_63650_3450_25', '19B001_63675_3400_25', '19B001_63675_3425_25', '19B001_63675_3450_25', '19B001_63700_3425_25', '19B001_63700_3400_25', '19B001_63700_3450_25', '19B001_63725_3400_25', '19B001_63725_3425_25', '19B001_63725_3450_25'],

    'DemMosaicZone9.shp':['19A017_62350_5000_25', '19A017_62350_5050_25', '19A017_62350_5025_25', '19A017_62350_5075_25', '19A017_62375_5000_25', '19A017_62375_5025_25', '19A017_62375_5050_25', '19A017_62375_5075_25', '19A017_62400_5000_25', '19A017_62400_5025_25', '19A017_62400_5050_25', '19A017_62400_5075_25']}
    
    print(buffer_path)
    for zone in zones:
        polylines = gpd.read_file(os.path.join(gt_path,zone))
        print(zone)
        for tile in zones[zone]:
            polygon_values = []
            try:
                polygons = gpd.read_file(os.path.join(buffer_path,tile+'.shp'))
                print(tile)
                
                for idx, polygon in polygons.iterrows():  
                    intersecting_segments = polylines.intersection(polygon.geometry)
                    valid_segments = intersecting_segments[~intersecting_segments.is_empty]
                    
                    if not valid_segments.empty:    
                        valid_segments = gpd.GeoSeries(valid_segments)        
                        segment_lengths = valid_segments.length        
                        polyline_attributes = polylines.loc[valid_segments.index, 'id']        
                        weighted_value = (segment_lengths * polyline_attributes).sum() / segment_lengths.sum()
                        if weighted_value <= 1.5:
                            polygon_values.append(1)
                        else:
                            polygon_values.append(2)
                        
                    else:
                        polygon_values.append(int(0))
                

                polygons['id'] = polygon_values

                polygons.to_file(os.path.join(output_path,tile+'.shp'))
            except:
                pass
            
if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='uses the buffer to sample the class of the channels based on the ground truth vectors',
                                     add_help=True,formatter_class=argparse.HelpFormatter)
    parser.add_argument('buffer_path', help='path to the buffered polylines')
    parser.add_argument('output_path',help='path to the output sampled buffer')
    parser.add_argument('gt_path',help='path to the ground truth data (vector)')
    args = vars(parser.parse_args())
    class_sampling(**args)
