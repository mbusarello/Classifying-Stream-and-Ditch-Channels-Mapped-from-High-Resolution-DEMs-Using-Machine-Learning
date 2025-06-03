"""
@author: Mariana Busarello, 2025
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point,LineString
from shapely.ops import nearest_points
import os
import argparse

def connecting_inference(input_path,output_path):
    for shp in os.listdir(input_path):
        if shp.endswith('.shp'):  
            input_shapefile = os.path.join(input_path,shp)
            polylines = gpd.read_file(input_shapefile)
            
            points = []
            for index, row in polylines.iterrows():
                line = row.geometry
                fid = row['FID'] if 'FID' in polylines.columns else index
                
                if line.geom_type == 'LineString':
                    start_point = Point(line.coords[0])
                    end_point = Point(line.coords[-1])
            
                    points.append({'geometry': start_point, 'type': 'start', 'FID': fid})
                    points.append({'geometry': end_point, 'type': 'end', 'FID': fid})
                    
                elif line.geom_type == 'MultiLineString':
                    for part in line:
                        start_point = Point(part.coords[0])
                        end_point = Point(part.coords[-1])
                        
                        points.append({'geometry': start_point, 'type': 'start', 'FID': fid})
                        points.append({'geometry': end_point, 'type': 'end', 'FID': fid})
            
            endpoints_gdf = gpd.GeoDataFrame(points, crs=polylines.crs)
            
            vertices = []
            for index, row in polylines.iterrows():
                line = row.geometry
                fid = row['FID'] if 'FID' in polylines.columns else index
            
                if line.geom_type == 'LineString':
                    for coord in line.coords:
                        vertices.append({'geometry': Point(coord), 'FID': fid})
                elif line.geom_type == 'MultiLineString':
                    for part in line:
                        for coord in part.coords:
                            vertices.append({'geometry': Point(coord), 'FID': fid})
            
            vertices_gdf = gpd.GeoDataFrame(vertices, crs=polylines.crs)
            
            line_segments = []
            for index, endpoint_row in endpoints_gdf.iterrows():
                endpoint = endpoint_row.geometry
                endpoint_fid = endpoint_row['FID']
            
                buffer = endpoint.buffer(10)
            
                nearby_vertices = vertices_gdf[vertices_gdf.geometry.within(buffer)]
                nearby_vertices = nearby_vertices[nearby_vertices['FID'] != endpoint_fid]
            
                if not nearby_vertices.empty:
                    nearest_geom = nearest_points(endpoint, nearby_vertices.geometry.unary_union)[1]
                    nearest_vt = nearby_vertices[nearby_vertices.geometry == nearest_geom].iloc[0]
            
                    
                    distance = endpoint.distance(nearest_geom)
                    if distance <= 10:
                        line_segment = LineString([endpoint, nearest_geom])
                        line_segments.append({
                            'geometry': line_segment,
                            'FID_ed': endpoint_fid,
                            'FID_vt': nearest_vt['FID'],
                            'type': endpoint_row['type']
                        })
            
            lines_gdf = gpd.GeoDataFrame(line_segments, crs=polylines.crs)
            lines_gdf['distance'] = lines_gdf.geometry.length
            con_sorted = lines_gdf.sort_values(by=['FID_ed', 'FID_vt', 'distance'], ascending=[True, True, True])
            con_clean1 = con_sorted.drop_duplicates(subset=['FID_ed', 'FID_vt'], keep='first')
            con_clean1['FID_ep_s'], con_clean1['FID_v_s'] = zip(*con_clean1[['FID_ed', 'FID_vt']].apply(lambda x: sorted(x), axis=1))
            con_clean1_sorted = con_clean1.sort_values(by=['distance'])
            con_clean1_result = con_clean1_sorted.drop_duplicates(subset=['FID_ep_s', 'FID_v_s'], keep='first')
            
            con_clean = con_clean1_result[['FID_ed', 'FID_vt', 'geometry']]
            all_lines = pd.concat([polylines, con_clean], ignore_index=True)
            all_lines = gpd.GeoDataFrame(all_lines, geometry='geometry', crs=polylines.crs)
            last_value = all_lines['FID'].dropna().iloc[-1]
            last_value = int(last_value)
            nan_indices = all_lines['FID'][all_lines['FID'].isna()].index
            sequential_fill_values = range(last_value + 1, last_value + 1 + len(nan_indices))
            all_lines.loc[nan_indices, 'FID'] = sequential_fill_values
            
            all_lines['FID'] = all_lines['FID'].astype(int)
            all_lines.to_file(os.path.join(output_path,shp))

if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='connects the gaps between the inference lines',
                                     add_help=True,formatter_class=argparse.HelpFormatter)
    parser.add_argument('input_path', help='path to the predicted vector lines to be connected')
    parser.add_argument('output_path', help='destination folder of the connected shapefiles')
    args = vars(parser.parse_args())
    connecting_inference(**args)