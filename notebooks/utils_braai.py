from astropy.io import fits
from astropy.visualization import ZScaleInterval
from bson.json_util import loads, dumps
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


def plot_error_comp(model, y_true, y_probs, thresholds=[0.5]):
    """
    Plot the FNR and FPR as a Function of Threshold
    """

    preds_bogus = y_probs[y_true == 0]
    preds_real  = y_probs[y_true == 1]
    rbbins = np.arange(-0.0001, 1.0001, 0.0001)

    h_b, e_b = np.histogram(preds_bogus, bins=rbbins, density=True)
    h_b_c = np.cumsum(h_b)
    h_r, e_r = np.histogram(preds_real, bins=rbbins, density=True)
    h_r_c = np.cumsum(h_r)
    
    # Tasas y Curvas
    fnr_curve = h_r_c / np.max(h_r_c)
    fpr_curve = 1 - h_b_c / np.max(h_b_c)
    mmce = (fnr_curve + fpr_curve) / 2
    rb_thres = np.array(list(range(len(h_b)))) / len(h_b)
    
    fig = plt.figure(figsize=(7, 4), dpi=100)
    ax = fig.add_subplot(111)
    
    ax.plot(rb_thres, fnr_curve, label='False Negative Rate (FNR)', linewidth=1.5)
    ax.plot(rb_thres, fpr_curve, label='False Positive Rate (FPR)', linewidth=1.5)
    ax.plot(rb_thres, mmce, '--', label='Mean misclassification error', color='gray', linewidth=1.5)
    
    ax.set_xlim([-0.05, 1.05])
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_yscale('log')
    ax.set_ylim([5e-4, 1])
    
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.1%}'.format(x) if x < 0.01 else '{:,.0%}'.format(x) for x in vals])
    t = thresholds[0]
    m_t = rb_thres < t
        
    fnr = np.array(fnr_curve)[m_t][-1]
    fpr = np.array(fpr_curve)[m_t][-1]
        
    print(f"Th: {t}, FNR: {fnr*100:.4f}%, FPR: {fpr*100:.4f}%")
        
    ax.vlines(t, 0, max(fnr, fpr), color='black')
    ax.text(t - .05, max(fnr, fpr) + 0.01, f' {fnr*100:.1f}% FNR\n {fpr*100:.1f}% FPR', fontsize=10)


    ax.set_xlabel('RB score threshold')
    ax.set_ylabel('Cumulative percentage')
    ax.legend(loc='lower center')
    ax.grid(True, which='major', linewidth=.5)
    ax.grid(True, which='minor', linewidth=.3)

    plt.title(f"{model} - FNR and FPR as a Function of Threshold")
    plt.tight_layout()
    plt.show()





"""
The functions in this file were adapted from Dmitry Duev's `braai` project.

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