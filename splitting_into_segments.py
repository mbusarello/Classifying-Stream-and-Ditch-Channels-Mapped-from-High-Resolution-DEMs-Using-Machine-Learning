"""
@author: Mariana Busarello, 2025
"""

import os
import geopandas as gpd
from shapely.geometry import LineString
import numpy as np
import argparse

def distance(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def interpolate_point(p1, p2, t):
    return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))

def split_polyline(vertices, max_length):
    new_segments = []
    current_segment = []
    current_length = 0

    for i in range(1, len(vertices)):
        p1 = vertices[i - 1]
        p2 = vertices[i]
        current_segment.append(p1)

        segment_length = distance(p1, p2)

        while current_length + segment_length > max_length:
            remaining_length = max_length - current_length
            t = remaining_length / segment_length
            split_point = interpolate_point(p1, p2, t)
            current_segment.append(split_point)
            new_segments.append(LineString(current_segment))


            current_segment = [split_point]
            current_length = 0
            p1 = split_point
            segment_length = distance(p1, p2)

        current_length += segment_length

    current_segment.append(vertices[-1])
    new_segments.append(LineString(current_segment))

    return new_segments

def process_shapefile(input_path, output_path, split_length):
    gdf = gpd.read_file(input_path)
    gdf = gdf.set_crs(epsg=3006)
    split_geometries = []
    
    for idx, row in gdf.iterrows():
        geom = row.geometry

        if geom.geom_type == 'LineString':
            vertices = list(geom.coords)
            split_lines = split_polyline(vertices, split_length)
            split_geometries.extend(split_lines)

        elif geom.geom_type == 'MultiLineString':
            for line in geom:
                vertices = list(line.coords)
                split_lines = split_polyline(vertices, split_length)
                split_geometries.extend(split_lines)


    split_gdf = gpd.GeoDataFrame(geometry=split_geometries, crs=gdf.crs)
    split_gdf.to_file(output_path)

def split_shapefile(input_path,output_path,split_length):
    for shape in os.listdir(input_path):
        if shape.endswith('.shp'):            
            input_shapefile = os.path.join(input_path,shape)
            output_shapefile = os.path.join(output_path,shape)
            process_shapefile(input_shapefile, output_shapefile, int(split_length))

if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='splits the channels polylines into the desired length',
                                     add_help=True,formatter_class=argparse.HelpFormatter)
    parser.add_argument('input_path', help='path to breached vectorized streams')
    parser.add_argument('output_path', help='destination path')
    parser.add_argument('split_length', help='length to split')
    args = vars(parser.parse_args())
    split_shapefile(**args) 