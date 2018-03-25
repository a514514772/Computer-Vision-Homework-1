import cv2
import numpy as np
import scipy.ndimage as ndimg
import matplotlib.pyplot as plt

from scipy import misc
from scipy import ndimage

import math

def GLPF(sigma, m, n):
    cx = m // 2 + 1
    cy = n // 2 + 1 
    ofilter = np.zeros((m, n)).astype(np.float32)
    for x in range(m):
        for y in range(n):
            dis = (cx - x) ** 2 + (cy - y) ** 2
            ofilter[x, y] = np.exp(-dis / (2 * sigma * sigma))
    return ofilter

def GHPF(sigma, m, n):
    cx = m // 2 + 1
    cy = n // 2 + 1
    ofilter = np.zeros((m, n)).astype(np.float32)
    for x in range(m):
        for y in range(n):
            dis = (cx - x) ** 2 + (cy - y) ** 2
            ofilter[x, y] = 1 - np.exp(-dis / (2 * sigma * sigma))
    return ofilter

def fft_filtering(input_img, f):
    dft = np.fft.fft2(input_img)
    dft = np.fft.fftshift(dft)
    
    mag_input = 20*np.log(np.abs(dft)+1)
    
    fft_output = dft * f 
    output = np.fft.ifft2(np.fft.ifftshift(fft_output)).real
    
    mag_output = 20*np.log(np.abs(fft_output)+1)

    return mag_input, mag_output, output

def combine_all_channels(info):
    info = np.concatenate([info])
    info = np.transpose(info, (1, 2, 0))
    return info