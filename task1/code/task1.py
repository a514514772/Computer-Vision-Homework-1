import argparse
import cv2
import numpy as np
import scipy.ndimage as ndimg
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="CV Task1 hybrid images")
parser.add_argument('-himg', type=str, default='../data/cat.bmp', help='the path to a high frequency image')
parser.add_argument('-limg', type=str, default='../data/dog.bmp', help='the path to a low frequency image')
parser.add_argument('-output', type=str, default='./', help='the output path')

args = parser.parse_args()

plt.close()

def centeralize_img(img):
	output = np.copy(img)
	for i in range(output.shape[0]):
		for j in range(output.shape[1]):
			output[i, j, :] *= ((-1) ** (i+j))

	return output

himg = plt.imread(args.himg, format='bmp')/255.
print ("shape of the high freq. image: ", himg.shape)
limg = plt.imread(args.limg, format='bmp')/255.
print ("shape of the low freq. image: ", limg.shape)

limg = centeralize_img(limg)
low_fft_shift = np.fft.fft2(limg)
low_fft_shift = np.fft.fftshift(low_fft_shift)
magnitude_spectrum = 20*np.log(np.abs(low_fft_shift))

print (magnitude_spectrum.shape)

'''
blur_kernel = np.ones((5, 5)) / 25
output = cv2.filter2D(limg, -1, blur_kernel)
'''

#plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.figure()
plt.imshow(magnitude_spectrum[:, :, 0], cmap='gray')
plt.show()