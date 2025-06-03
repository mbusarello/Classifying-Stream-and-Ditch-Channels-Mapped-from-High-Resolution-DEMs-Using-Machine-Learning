"""
@author: Mariana Busarello, 2025
"""

import os
import numpy as np
from skimage.morphology import skeletonize
from osgeo import gdal, osr
import rasterio
import traceback
import whitebox
whitebox.download_wbt(linux_musl=True, reset=True)
wbt = whitebox.WhiteboxTools()
wbt.verbose = False

def raster_to_line(img_name, vector_lines):
    wbt.raster_to_vector_lines(
        i=img_name,
        output=vector_lines
    )

def write_gtiff(array, gdal_obj, outputpath, dtype=gdal.GDT_Byte, options=0, color_table=0, nbands=1, nodata=False):
    gt = gdal_obj.GetGeoTransform()
    width = np.shape(array)[1]
    height = np.shape(array)[0]
    driver = gdal.GetDriverByName("GTiff")
    if options != 0:
        dest = driver.Create(outputpath, width, height, nbands, dtype, options)
    else:
        dest = driver.Create(outputpath, width, height, nbands, dtype)
    if dest is None:
        print(f"Failed to create destination file: {outputpath}")
        return

    if color_table != 0:
        dest.GetRasterBand(1).SetColorTable(color_table)
    if len(array.shape) == 3:
        if nbands == 1:

            array = array[:, :, 0]
        band = dest.GetRasterBand(1)
        if band is not None:
            band.WriteArray(array)
        else:
            print("Failed to get band 1 for writing.")
    else:  
        band = dest.GetRasterBand(1)
        if band is not None:
            band.WriteArray(array)
        else:
            print("Failed to get band 1 for writing.")
    if nodata is not False:
        dest.GetRasterBand(1).SetNoDataValue(nodata)

    dest.SetGeoTransform(gt)
    wkt = gdal_obj.GetProjection()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    dest.SetProjection(srs.ExportToWkt())

    dest = None
    print(f"File {outputpath} written successfully.")

def main(input_path, out_path_binary, img_type):
    if not os.path.exists(input_path):
        raise ValueError('Input path does not exist: {}'.format(input_path))
    if os.path.isdir(input_path):
        imgs = [os.path.join(input_path, f) for f in os.listdir(input_path)
                if f.endswith('.tif')]
    else:
        imgs = [input_path]

    for img_path in imgs:
        try:
            with rasterio.open(img_path) as src:
                binary_image = src.read(1)

            skeleton = skeletonize(binary_image.astype(bool))

            img_name = os.path.basename(img_path).split('.')[0]
            InutFileWithKnownExtent = gdal.Open(img_path)
            raster_name = os.path.join(out_path_binary, '{}.{}'.format(img_name, img_type))
            write_gtiff(skeleton.astype(np.uint8), InutFileWithKnownExtent, raster_name)

            vector_name = os.path.join(out_path_binary, '{}.{}'.format(img_name, 'shp'))
            raster_to_line(raster_name, vector_name)

        except Exception as e:
            print('Failed to run processing')
            print(f"Error: {e}")
            traceback.print_exc()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                       description='Creates skelletonized lines from the deep learning inference',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_path', help='path to the inference raster file')
    parser.add_argument('out_path_binary', help='Path to output raster and shapefile with skelletonized lines')
    parser.add_argument('--img_type', help='filetype for the raster file (recommended: TIF)')
    args = vars(parser.parse_args())
    main(**args)
