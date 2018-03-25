import argparse
import cv2
import numpy as np
import scipy.ndimage as ndimg
import matplotlib.pyplot as plt

from scipy import misc
from scipy import ndimage

import math

from utils import GLPF, GHPF, fft_filtering, combine_all_channels

parser = argparse.ArgumentParser(description="CV Task1 hybrid images")
parser.add_argument('-himg', type=str, default='../data/einstein.bmp', help='the path to a high frequency image')
parser.add_argument('-limg', type=str, default='../data/marilyn.bmp', help='the path to a low frequency image')
parser.add_argument('-output', type=str, default='./hybird.png', help='the output path')

args = parser.parse_args()

plt.close()

himg = ndimage.imread(args.himg, flatten=False)
print ("shape of the high freq. image: ", himg.shape)
limg = ndimage.imread(args.limg, flatten=False)
print ("shape of the low freq. image: ", limg.shape)

gf = GLPF(10, limg.shape[0], limg.shape[1])
hf = GHPF(25, limg.shape[0], limg.shape[1])

if len(limg.shape) == 3:
    # Deal with RGB images
    lmagnitude_spectrum, lmag_output, loutput = [], [], []
    hmagnitude_spectrum, hmag_output, houtput = [], [], []
    
    # FFT transform of RGB channels respectively
    for i in range(3):
        lm_s, lm_out, lout = fft_filtering(limg[:, :, i], gf)
        hm_s, hm_out, hout = fft_filtering(himg[:, :, i], hf)
        lmagnitude_spectrum.append(lm_s)
        lmag_output.append(lm_out)
        loutput.append(lout)
        hmagnitude_spectrum.append(hm_s)
        hmag_output.append(hm_out)
        houtput.append(hout)
    
    # concatenate results from three FFT to form a single matrix
    lmagnitude_spectrum, lmag_output, loutput = combine_all_channels(lmagnitude_spectrum), \
        combine_all_channels(lmag_output), \
        combine_all_channels(loutput)
    hmagnitude_spectrum, hmag_output, houtput = combine_all_channels(hmagnitude_spectrum), \
        combine_all_channels(hmag_output), \
        combine_all_channels(houtput)
        
    # generate hybrid image
    hybrid = (loutput + houtput)
    
else: 
    # Deal with gray-scale images
    lmagnitude_spectrum, lmag_output, loutput = fft_filtering(limg, gf)
    hmagnitude_spectrum, hmag_output, houtput = fft_filtering(himg, hf)
    hybrid = loutput + houtput
    
misc.imsave("hybrid.png", hybrid.real)

# post-process for the numerical issues
hybrid[hybrid<0] = 0
hybrid /= 255.

plt.figure(dpi=300)
plt.subplot(131),plt.imshow(himg, cmap = 'gray')
plt.title('High-pass Input'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(limg, cmap = 'gray')
plt.title('Low-pass Input'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(hybrid, cmap = 'gray')
plt.title('Output Image'), plt.xticks([]), plt.yticks([])
plt.savefig("output.png")
#plt.show()