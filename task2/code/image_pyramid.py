import numpy as np
import cv2
from matplotlib import pyplot as plt

def fft(img):
    # If the image is a color image, convert the image into one channel
    if img.ndim > 2 :
        # Since the default value of the program is in RGB
        # it's converting RGB to GRAY
        if img.shape[2] > 1:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Let's convert to FFT
    dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

    return magnitude_spectrum

def generate_gaussian_pyramid(img, nPyramidLayer=4, gaussianKernelSize=5, gaussianKernelSigma=2):
    if (nPyramidLayer < 2):
        raise ValueError("Image Pyramid must have minimum 2 layer.")

    # Variable Initialization
    lower_res = img.copy()
    gPyramid = [lower_res]

    # Generate 2D Gaussian Kernel based on input
    gk = cv2.getGaussianKernel(gaussianKernelSize, gaussianKernelSigma)
    gk = np.dot(gk, gk.transpose())

    # Generate Gaussian Pyramid Image List
    for x in range(1, nPyramidLayer):
        # If the current pixel is less than 2 pixel, then stop downsampling.
        if lower_res.shape[0] < 2 or lower_res.shape[1] < 2:
            break
        # Do Gaussian Smoothing
        lower_res = cv2.filter2D(lower_res, -1, gk)
        # Downsampling using the NEAREST neighbor interpolation
        lower_res = cv2.resize(lower_res, ((lower_res.shape[1] + 1) / 2, (lower_res.shape[0] + 1) / 2),
                               interpolation=cv2.INTER_NEAREST)
        # Store result in the array
        gPyramid.append(lower_res)

    return gPyramid

def generate_laplacian_pyramid(gPyramid, gaussianKernelSize=5, gaussianKernelSigma=2):
    # The last layer image in gaussian pyramid is the same. G_N = L_N
    lpA = [gPyramid[len(gPyramid) - 1]]

    # Generate Gaussian Kernel based on input
    gk = cv2.getGaussianKernel(gaussianKernelSize, gaussianKernelSigma)
    gk = np.dot(gk, gk.transpose())

    # Generate Laplacian Pyramid List from the Gaussian Pyramid
    for i in xrange(len(gPyramid) - 1, 0, -1):
        # Upsampling to the size of the next G_N
        G = cv2.resize(gPyramid[i], (gPyramid[i - 1].shape[1], gPyramid[i - 1].shape[0]),
                       interpolation=cv2.INTER_NEAREST)

        # Do Some Gaussian Smoothing after Upsampling
        G = cv2.filter2D(G, -1, gk)

        # We Know that L_N = G_N - Smooth(Up(L_N-1))
        L = np.subtract(gPyramid[i - 1], G, dtype=float)

        # Normalization For the Greyish Effect. Such in Example
        L = ((L + 255) / (255 * 2)) * 255
        L = L.astype(np.uint8)

        # Store the result into an array
        lpA.append(L)

    return lpA

def show_result(gPyramid, lPyramid, output_filename="output.pdf"):
    # If this raises, then,
    # the code is wrong.
    if len(gPyramid) != len(lPyramid):
        raise ValueError("The length of Laplacian Pyramid and Gaussian Pyramid is not the same.")


    height = gPyramid[0].shape[0]   # the height of the original image
    width = gPyramid[0].shape[1]    # the width of the original image
    length = len(gPyramid)          # length of the pyramid

    # Variable Initialization
    fft_result_gPyramid = []
    fft_result_lPyramid = []
    fft_z_max = -99999
    fft_z_min = 999999

    #Iterate over the Gaussian Pyramid to obtain the FFT and display it
    for i, image in enumerate(gPyramid):
        plt.subplot(4, length, i + 1), plt.imshow(image, cmap="gray")
        plt.xticks([]), plt.yticks([])

        image = cv2.resize(image, (width, height))
        fft_result = fft(image)
        fft_z_max = max(fft_z_max, np.max(fft_result))
        fft_z_min = min(fft_z_min, np.min(fft_result))
        fft_result_gPyramid.append(fft_result)

    # Iterate over the Laplacian Pyramid to obtain the FFT and display it
    r_lPyramid = lPyramid[::-1]  # reverse the laplacian pyramid for visualization purposes.
    for i, image in enumerate(r_lPyramid):
        plt.subplot(4, length, length + i + 1), plt.imshow(image, cmap="gray")
        plt.xticks([]), plt.yticks([])

        image = cv2.resize(image, (width, height))
        fft_result = fft(image)
        fft_z_max = max(fft_z_max, np.max(fft_result))
        fft_z_min = min(fft_z_min, np.min(fft_result))
        fft_result_lPyramid.append(fft_result)

    # Visualize the FFT result
    for i in range(length):
        plt.subplot(4, length, length * 2 + i + 1), plt.imshow(fft_result_gPyramid[i], vmin=fft_z_min, vmax=fft_z_max,
                                                               cmap='jet')
        plt.xticks([]), plt.yticks([])
        plt.subplot(4, length, length * 3 + i + 1), plt.imshow(fft_result_lPyramid[i], vmin=fft_z_min, vmax=fft_z_max,
                                                               cmap='jet')
        plt.xticks([]), plt.yticks([])

    # Save and Show the plot.
    fig = plt.gcf()
    fig.canvas.set_window_title('Image Pyramid Result')
    plt.show()
    # fig.savefig(output_filename)


def generate_pyramid_layout(gPyramid):
    if len(gPyramid) == 0:
        return ValueError("List is empty")
    elif len(gPyramid) == 1:
        return gPyramid[0]

    default_size = gPyramid[1].shape
    gImage = gPyramid[1].copy()
    height_cnt = gPyramid[1].shape[0]

    for image in gPyramid[2:]:
        _size = list(default_size)
        _size[0] = image.shape[0]
        gImage = np.append(gImage, np.zeros(tuple(_size), dtype=int), axis=0)
        gImage[height_cnt:, :image.shape[1]] = image
        height_cnt += image.shape[0]

    _size = list(default_size)
    _size[0] = max(gImage.shape[0], gPyramid[0].shape[0])

    result = np.append(gPyramid[0], np.zeros(tuple(_size), dtype=int), axis=1)
    result[:gImage.shape[0], gPyramid[0].shape[1]:gPyramid[0].shape[1] + gImage.shape[1]] = gImage
    return result