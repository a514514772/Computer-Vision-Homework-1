import cv2
import numpy as np

def get_gradient(image_file) :
    X_way_gradients = cv2.Sobel(image_file,cv2.CV_32F,1,0,ksize=3)
    Y_way_gradients = cv2.Sobel(image_file,cv2.CV_32F,0,1,ksize=3)
    two_gradients_combine = cv2.addWeighted(np.absolute(X_way_gradients), 0.5, np.absolute(Y_way_gradients), 0.5, 0)
    return two_gradients_combine


if __name__ == '__main__':
    image_file =  cv2.imread("../data/cathedral.jpg", cv2.IMREAD_GRAYSCALE);

    picture_size = image_file.shape
    #print (picture_size)
    height = int(picture_size[0] / 3);
    width = picture_size[1]

    im_color = np.zeros((height,width,3), dtype=np.uint8 )
    for i in range(0,3) :
        im_color[:,:,i] = image_file[ i * height:(i+1) * height,:]

    im_aligned = np.zeros((height,width,3), dtype=np.uint8 )

    im_aligned[:,:,2] = im_color[:,:,2]

    motion_mode = cv2.MOTION_HOMOGRAPHY

    warp_matrix = np.eye(3, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000,  1e-10)
    for i in range(0,2) :
        (cc, warp_matrix) = cv2.findTransformECC (get_gradient(im_color[:,:,2]), get_gradient(im_color[:,:,i]),warp_matrix, motion_mode, criteria)
        im_aligned[:,:,i] = cv2.warpPerspective (im_color[:,:,i], warp_matrix, (width,height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    cv2.imshow("Aligned Image", im_aligned)
    cv2.imwrite('../data/cathedral-RGB.jpg',im_aligned)