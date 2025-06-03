#from https://github.com/williamlidberg/Detection-of-hunting-pits-using-airborne-laser-scanning-and-deep-learning
import tifffile as tiff
import numpy as np
import sys
sys.path
from osgeo import ogr
from osgeo import gdal
from osgeo import osr


def write_gtiff(array, georef, out_path):
    # Ensure array is correctly structured
    if array.ndim not in [2, 3]:
        raise ValueError("Array must be 2D or 3D.")

    # Get array dimensions
    if array.ndim == 3:
        height, width, num_classes = array.shape
    else:
        height, width = array.shape
        num_classes = 1

    driver = gdal.GetDriverByName('GTiff')
    if driver is None:
        raise ValueError("GDAL driver GTiff is not available.")

    dest = driver.Create(out_path, width, height, num_classes, gdal.GDT_Float32)
    if dest is None:
        raise ValueError(f"Could not create GeoTIFF file at the specified path: {out_path}")

    # Set the georeferencing information
    dest.SetGeoTransform(georef.GetGeoTransform())
    dest.SetProjection(georef.GetProjectionRef())

    # If the array is 3D (multiple classes), we need to write each band
    if num_classes > 1:
        for i in range(num_classes):
            out_band = dest.GetRasterBand(i + 1)
            out_band.WriteArray(array[:, :, i])  # Write probability map for class i
            out_band.FlushCache()
    else:
        # If only one class, we treat the array as 2D
        dest.GetRasterBand(1).WriteArray(array)
        
    dest = None
