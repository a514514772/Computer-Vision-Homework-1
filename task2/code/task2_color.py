import cv2
import image_pyramid as ip
from matplotlib import pyplot as plt

def read_file(file_name):
    img = cv2.imread(file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def show_image(img, title):
    plt.imshow(img)
    plt.title(title)
    plt.xticks([]), plt.yticks([])
    fig = plt.gcf()
    fig.canvas.set_window_title(title)
    plt.show()

img = read_file("../data/dog.bmp")
show_image(img, "Input Image")

gPyramid = ip.generate_gaussian_pyramid(img, 5)
lPyramid = ip.generate_laplacian_pyramid(gPyramid)
show_image(ip.generate_pyramid_layout(gPyramid), "Gaussian Image Pyramid")
show_image(ip.generate_pyramid_layout(lPyramid[::-1]), "Laplacian Image Pyramid")

ip.show_result(gPyramid, lPyramid)