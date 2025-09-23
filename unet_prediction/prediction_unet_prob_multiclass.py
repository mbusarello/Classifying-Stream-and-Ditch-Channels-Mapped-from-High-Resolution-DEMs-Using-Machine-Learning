#from https://github.com/williamlidberg/Detection-of-hunting-pits-using-airborne-laser-scanning-and-deep-learning/tree/main

import os
import numpy as np
import tifffile
from osgeo import gdal
import utils.unet
import utils.WriteGeotiff
#import tensorflow as tf
#tf.config.threading.set_intra_op_parallelism_threads(1)
#tf.config.threading.set_inter_op_parallelism_threads(1)


def patchify_x(img, start_y, patches, tile_size, margin, width, channel_last):
    start_x = 0
    while start_x + tile_size <= width:
        if channel_last:
            patches.append(img[start_y:start_y+tile_size,
                               start_x:start_x+tile_size, :].copy())
        else:
            patches.append(img[:, start_y:start_y+tile_size,
                               start_x:start_x+tile_size].copy())

        start_x += tile_size - 2 * margin

    if start_x < width:
        start_x = width - tile_size
        if channel_last:
            patches.append(img[start_y:start_y+tile_size,
                               start_x:start_x+tile_size, :].copy())
        else:
            patches.append(img[:, start_y:start_y+tile_size,
                               start_x:start_x+tile_size].copy())


def patchify(img, tile_size, margin, channel_last):
    patches = []

    if channel_last:
        height, width, _ = img.shape
    else:
        _, height, width = img.shape
    start_y = 0
    while start_y + tile_size <= height:
        patchify_x(img, start_y, patches, tile_size, margin, width,
                   channel_last)
        start_y += tile_size - 2 * margin

    if start_y < height:
        start_y = height - tile_size
        patchify_x(img, start_y, patches, tile_size, margin, width,
                   channel_last)

    return patches


def start_and_end(base, tile_size, margin, limit, remainder):
    if base == 0:
        src_start = 0
        src_end = tile_size - margin
    elif base + (tile_size - margin) > limit:
        src_start = tile_size - remainder
        src_end = tile_size
    else:
        src_start = margin
        src_end = tile_size - margin

    return src_start, src_end


def unpatchify(shape, patches, tile_size, margin):
    """
    Reconstructs full-size prediction from patches.

    Parameters
    ----------
    shape : tuple
        (height, width, channels) if channel_last, otherwise (height, width).
    patches : list of tuples
        Each element is (patch_class, patch_prob), where:
          - patch_class : (tile_size, tile_size) int array
          - patch_prob  : (tile_size, tile_size, num_classes) float array
    tile_size : int
        Size of each patch.
    margin : int
        Overlap margin between patches.

    Returns
    -------
    img_class : np.ndarray
        (H, W) predicted class labels.
    img_prob : np.ndarray
        (H, W, num_classes) probability maps.
    """
    height, width = shape[:2]
    num_classes = patches[0][1].shape[-1]

    img_class = np.zeros((height, width), dtype=np.int32)
    img_prob = np.zeros((height, width, num_classes), dtype=np.float32)

    dest_start_y = 0
    dest_start_x = 0

    for patch_class, patch_prob in patches:
        remain_width = width - dest_start_x
        remain_height = height - dest_start_y

        src_start_y, src_end_y = start_and_end(
            dest_start_y, tile_size, margin, height, remain_height
        )
        src_start_x, src_end_x = start_and_end(
            dest_start_x, tile_size, margin, width, remain_width
        )

        y_length = src_end_y - src_start_y
        x_length = src_end_x - src_start_x


        img_class[
            dest_start_y:dest_start_y + y_length,
            dest_start_x:dest_start_x + x_length
        ] = patch_class[src_start_y:src_end_y, src_start_x:src_end_x]


        img_prob[
            dest_start_y:dest_start_y + y_length,
            dest_start_x:dest_start_x + x_length, :
        ] = patch_prob[src_start_y:src_end_y, src_start_x:src_end_x, :]


        dest_start_x += x_length
        if dest_start_x >= width:
            dest_start_x = 0
            dest_start_y += y_length

    return img_class, img_prob



def read_input(bands, channel_last):
    '''Assemble input from list of provided tif files
       inputs will be added in order in which they are provided

    Parameters
    ----------
    bands : list of pathes to tif files
    channel_last : indicate location of channel dimension

    Returns
    -------
    Tensor of shape (input height, input width, number of bands) or (number of
    bands, input height, input width) - depending on channel_last

    '''
    tmp = tifffile.imread(bands[0])
    if channel_last:
        img = np.zeros([*tmp.shape, len(bands)])
    else:
        img = np.zeros([len(bands), *tmp.shape])
    for i, band in enumerate(bands):
        tmp = tifffile.imread(band)
        tmp = tmp.astype(np.float32)
        if channel_last:
            img[:, :, i] = tmp
        else:
            img[i, :, :] = tmp

    return img



def main(img_path, model_path, out_path, model_type, band_wise, depth,
         img_type, tile_size, margin, classes):


    for path in img_path:
        if not os.path.exists(path):
            raise ValueError(f'Input path does not exist: {path}')


    if os.path.isdir(img_path[0]):
        imgs = []
        for path in img_path:
            tmp = [os.path.join(path, f) for f in os.listdir(path)
                   if not f.startswith('._') and f.endswith('.tif')]
            imgs.append(tmp)
    else:
        imgs = [[f] for f in img_path]


    model_cls = utils.unet.MODELS[model_type]
    if model_cls.CHANNEL_LAST:
        input_shape = (tile_size, tile_size, len(imgs))
    else:
        input_shape = (len(imgs), tile_size, tile_size)

    model = model_cls(
        input_shape,
        depth=depth,
        classes=len(classes.split(',')),
        entry_block=not band_wise,
        weighting=utils.unet.SegmentationModelInterface.WEIGHTING.NONE
    )
    model.load_weights(model_path)

    num_classes = len(classes.split(','))


    for bands in zip(*imgs):
        predicted = []
        img = read_input(bands, model.CHANNEL_LAST)

        do_patchify = tile_size < img.shape[0]
        patches = patchify(img, tile_size, margin, model.CHANNEL_LAST) if do_patchify else [img]


        for i in [8, 4, 2, 1]:
            if len(patches) % i == 0:
                batch_size = i
                break


        for i in range(0, len(patches), batch_size):
            batch = np.array(patches[i:i+batch_size])
            batch = batch.reshape((batch_size, *input_shape))
            out = model.proba(batch)  # (B, tile, tile, C)

            for b in range(batch.shape[0]):
                output = out[b]  # (tile, tile, C)
                predicted_classes = np.argmax(output, axis=-1)      # (tile, tile)
                patch_prob = output                                 # (tile, tile, C)
                predicted.append((predicted_classes, patch_prob))


        if do_patchify:
            if model.CHANNEL_LAST:
                out_class, out_prob = unpatchify(img.shape, predicted, tile_size, margin)
            else:
                out_class, out_prob = unpatchify(img.shape[1:], predicted, tile_size, margin)
        else:
            out_class, out_prob = predicted[0]


        img_name = os.path.basename(bands[0]).split('.')[0]
        InutFileWithKnownExtent = gdal.Open(bands[0])


        utils.WriteGeotiff.write_gtiff(
            out_class,
            InutFileWithKnownExtent,
            os.path.join(out_path, f"{img_name}_class.{img_type}")
        )


        for class_idx in range(num_classes):
            class_prob_map = out_prob[..., class_idx]
            if class_prob_map.ndim > 2:
                class_prob_map = class_prob_map.squeeze()
            utils.WriteGeotiff.write_gtiff(
                class_prob_map,
                InutFileWithKnownExtent,
                os.path.join(out_path, f"{img_name}_class_{class_idx}_prob.{img_type}")
            )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                       description='Runs the prediction on given '
                                   'image(s) and return the probability maps',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-I', '--img_path', action='append', help='Add path '
                        'to input images (either path to single image or to '
                        'folder containing images)')
    parser.add_argument('model_path')
    parser.add_argument('out_path', help='Path to output folder')
    parser.add_argument('model_type', help='Segmentation model to use',
                        choices=list(utils.unet.MODELS.keys()))
    parser.add_argument('--band_wise', action='store_true',
                        help='Apply separable convolutions on input bands.')
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--img_type', help='Output image file ending',
                        default='tif')
    parser.add_argument('--classes', help='List of class labels in ground '
                        'truth - order needs to correspond to weighting order',
                        default='0,1,2')
    parser.add_argument('--tile_size', help='Tile size', type=int,
                        default=250)
    parser.add_argument('--margin', help='Margin', type=int, default=100)

    args = vars(parser.parse_args())
    main(**args)