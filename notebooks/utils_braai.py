"""
Some functions in this file (`make_triplet`, `load_model_helper`)
were adapted from Dmitry Duev's `braai` project.

MIT License

Copyright (c) 2019 Dmitry Duev

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
=============================================================================
"""

from astropy.io import fits
from astropy.visualization import ZScaleInterval
from bson.json_util import loads, dumps
import gzip
import io
import json
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.utils import normalize as tf_norm
import os

def load_model_helper(path, model_base_name):
    """
        Build keras model using json-file with architecture and hdf5-file with weights
    """
    with open(os.path.join(path, f'{model_base_name}.architecture.json'), 'r') as json_file:
        loaded_model_json = json_file.read()
    m = model_from_json(loaded_model_json)
    m.load_weights(os.path.join(path, f'{model_base_name}_weights.h5'))

    return m


def make_triplet(alert, normalize: bool = False, old_norm: bool = False, to_tpu: bool = False):
    """
        Feed in alert packet
    """
    cutout_dict = dict()

    for cutout in ('science', 'template', 'difference'):
        cutout_data = loads(dumps([alert[f'cutout{cutout.capitalize()}']]))[0]

        # unzip
        with gzip.open(io.BytesIO(cutout_data), 'rb') as f:
            with fits.open(io.BytesIO(f.read())) as hdu:
                data = hdu[0].data
                # replace nans with zeros
                cutout_dict[cutout] = np.nan_to_num(data)
                # normalize
                if normalize:
                    if not old_norm:
                        cutout_dict[cutout] /= np.linalg.norm(cutout_dict[cutout])
                    else:
                        # model <= d6_m7
                        cutout_dict[cutout] = tf_norm(cutout_dict[cutout])

        # pad to 63x63 if smaller
        shape = cutout_dict[cutout].shape
        #print(shape)
        if shape != (63, 63):
            # print(f'Shape of {candid}/{cutout}: {shape}, padding to (63, 63)')
            cutout_dict[cutout] = np.pad(cutout_dict[cutout], [(0, 63 - shape[0]), (0, 63 - shape[1])],
                                         mode='constant', constant_values=1e-9)

    triplet = np.zeros((63, 63, 3))
    triplet[:, :, 0] = cutout_dict['science']
    triplet[:, :, 1] = cutout_dict['template']
    triplet[:, :, 2] = cutout_dict['difference']
    
    if to_tpu:
        # Edge TPUs require additional processing
        triplet = np.rint(triplet * 128 + 128).astype(np.uint8).flatten()
    
    return triplet

def plot_triplet(tr, triplet):
    fig = plt.figure(figsize=(8, 2), dpi=120)
    ax1 = fig.add_subplot(131)
    ax1.axis('off')
    interval = ZScaleInterval()
    limits = interval.get_limits(triplet[:, :, 0])
    # norm=LogNorm()
    ax1.imshow(tr[:, :, 0], origin='upper', cmap=plt.cm.bone, vmin=limits[0], vmax=limits[1])
    ax1.title.set_text('Science')
    ax2 = fig.add_subplot(132)
    ax2.axis('off')
    limits = interval.get_limits(triplet[:, :, 1])
    ax2.imshow(tr[:, :, 1], origin='upper', cmap=plt.cm.bone, vmin=limits[0], vmax=limits[1])
    ax2.title.set_text('Reference')
    ax3 = fig.add_subplot(133)
    ax3.axis('off')
    limits = interval.get_limits(triplet[:, :, 2])
    ax3.imshow(tr[:, :, 2], origin='upper', cmap=plt.cm.bone, vmin=limits[0], vmax=limits[1])
    ax3.title.set_text('Difference')
    plt.show()