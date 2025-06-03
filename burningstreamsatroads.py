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
    from WBT.whitebox_tools import WhiteboxTools
    whitebox.download_wbt(linux_musl=True, reset=True)
    wbt = WhiteboxTools()
    
def burning(filledDEMs,streams_vector,roads_vector,output_path,width):
    for mosaic in os.listdir(filledDEMs):
        wbt.burn_streams_at_roads(dem=os.path.join(filledDEMs,mosaic),
                                  streams=streams_vector,
                                  roads=roads_vector,
                                  output=os.path.join(output_path,mosaic),
                                  width=width
                                  )
                                  

if __name__== '__main__':
    parser = argparse.ArgumentParser(
        description='Burns the streams at the intersection with roads',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filledDEMs', help='path to DEMs previously filled')     
    parser.add_argument('streams_vector', help='path to the polyline vector with the stream channels')
    parser.add_argument('roads_vector', help='path to the polyline vector with the roads')
    parser.add_argument('output_path', help='destination path of the burned DEMs')
    parser.add_argument('width', help='maximum road enbankment in map units')
    args = vars(parser.parse_args())
    burning(**args)