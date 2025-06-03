"""
@author: Mariana Busarello, 2025
"""

import os
import rasterio as rio
import numpy as np
import argparse
try:
    import whitebox
    whitebox.download_wbt(linux_musl=True, reset=True)
    wbt = whitebox.WhiteboxTools()
except:
    from WBT.whitebox_tools import WhiteboxTools
    whitebox.download_wbt(linux_musl=True, reset=True)
    wbt = WhiteboxTools()
    
def min_probability(input_path,output_path):
    for pr in os.listdir(input_path):
        if pr.endswith('_prob.tif'):
            print(pr)
            input_raster = os.path.join(input_path,pr)
            output_raster = os.path.join(output_path,pr)
            r_o = rio.open(input_raster)
            r_r = r_o.read(1)
            #r_r = r_r.astype(np.uint8)
        
            r_t = np.where(r_r > 0.5, r_r, 0)
            file_save = rio.open(
                            output_raster,
                           'w',
                           driver='GTiff',
                           height=r_r.shape[0],
                           width=r_r.shape[1],
                           count=1,
                           dtype=r_r.dtype,
                           crs=r_o.crs,
                           transform = rio.Affine(
                               r_o.profile['transform'][0],r_o.profile['transform'][1],
                               r_o.profile['transform'][2],r_o.profile['transform'][3],
                               r_o.profile['transform'][4],r_o.profile['transform'][5])
                           )
        
            file_save.write(r_t,1)
            file_save.close()
        
if __name__== '__main__':
    parser = argparse.ArgumentParser(
        description='updates the probability maps, only keeping the value of cells above 0.5, the rest becomes zero',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_path', help='path to the probability maps output by the inference code')     
    parser.add_argument('output_path', help='path to the probability maps with the new threshold')

    args = vars(parser.parse_args())
    min_probability(**args)