import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import errno

from skimage.color import rgb2gray
from typing import List

GRAYSCALE = 1
RBG = 2

RGB2YIQ_MATRIX = np.array([[0.299, 0.587, 0.114],
                           [0.596, -0.275, -0.321],
                           [0.212, -0.523, 0.311]])


def grad_img():
    x = np.hstack([np.repeat(np.arange(0, 50, 2), 10)[None, :], np.array([255] * 6)[None, :]])
    grad = np.tile(x, (256, 1))
    norm_grad = grad.astype(np.float64) / grad.max()
    # print(np.histogram(grad, bins=256)[0])
    # plt.imshow(grad, cmap='gray')
    # plt.savefig('grad_gray.png')
    # plt.show()
    # plt.imshow(grad)
    # plt.savefig('grad.png')
    # plt.show()
    return grad, norm_grad


#######################################################
## (1) Reading an image into a given representation  ##
#######################################################

def read_image(filename: str, representation: int):
    """
    This function reads an image into a given representation.
    :param filename: the filename of an image on disk (could be grayscale or RGB).
    :param representation: representation code, either 1 or 2 defining whether the
                           output should be a grayscale image (1) or an RGB image (2).
    :return: an image represented by a matrix of type np.float64 with
             intensities normalized to the range [0, 1].
    :rtype: array_like['float64']
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)
    # make sure the matrix type is np.float64.
    img = imageio.imread(filename).astype(np.float64)
    # normalize intensities to the range [0, 1].
    img /= img.max()
    return img if representation == RBG else rgb2gray(img)


##############################
## (2) Displaying an image  ##
##############################

def imdisplay(filename: str, representation: int) -> None:
    """
    This function opens a new figure and display the loaded image in the converted
    representation.
    :param filename: the filename of an image on disk.
    :param representation: representation code.
    :return: nothing.
    """
    cmap = None if representation == RBG else 'gray'
    fig, ax = plt.subplots()
    img = ax.imshow(read_image(filename, representation), cmap=cmap)

    plt.colorbar(img, orientation='horizontal')
    fig.tight_layout()
    fig.show()


#######################################################
## (3) Transforming an RGB image to YIQ color space  ##
#######################################################

def rgb2yiq(imRGB):
    """
    This function transforms an RGB image into the YIQ color space.
    :param imRGB: an RBG image.
    :type imRGB: array_like[shape=(.., .., 3), dtype='float64']
    :return: an YIQ image.
    :rtype: array_like[shape=(.., .., 3), dtype='float64']
    """
    # return np.dot(imRGB[:, :, :3], RGB2YIQ_MATRIX.T)
    return imRGB[:, :, :3] @ RGB2YIQ_MATRIX.T


def yiq2rgb(imYIQ):
    """
    This function transforms an image from the YIQ color space into the RGB one.
    :param imYIQ: an YIQ image.
    :type imYIQ: array_like[shape=(.., .., 3), dtype='float64']
    :return: an RBG image.
    :rtype: array_like[shape=(.., .., 3), dtype='float64']
    """
    # return np.dot(imYIQ[:, :, :3], np.linalg.inv(RGB2YIQ_MATRIX).T)
    return imYIQ[:, :, :3] @ np.linalg.inv(RGB2YIQ_MATRIX).T


#################################
## (4) Histogram equalization  ##
#################################

def histogram_equalize(im_orig: np.ndarray) -> List:
    """
    This Function performs histogram equalization of a given grayscale or RGB image.
    :param im_orig: is the input grayscale or RGB float64 image with values in [0, 1].
    :type im_orig: array_like[dtype='float64']
    :return:a list [im_eq, hist_orig, hist_eq] where
            im_eq - is the equalized image. grayscale or RGB float64 image with values in [0, 1].
            hist_orig - is a 256 bin histogram of the original image (array with shape (256,) ).
            hist_eq - is a 256 bin histogram of the equalized image (array with shape (256,) ).
    """
    is_rgb = im_orig.ndim == 3
    # get the corresponding Y channel of the RGB image
    if is_rgb:
        im_orig = rgb2yiq(im_orig)[:, :, 0]
    im_orig *= 255  # just affects the values of the bins edges [0, 1] or [0, 255]
    im_orig = im_orig.astype(np.uint8)
    # The image histogram.
    hist_orig = np.histogram(im_orig, bins=256)[0]
    print(hist_orig)
    # The cumulative histogram.
    cum_hist = hist_orig.cumsum()
    # print(cum_hist)
    # The normalized cumulative histogram.
    # let m be the first gray level for which C(m) != 0
    m = np.argmax(cum_hist != 0)
    # print(m)
    cum_hist = np.around(255 * (cum_hist - cum_hist[m]) / (cum_hist[255] - cum_hist[m]))
    # print(cum_hist)
    # Map the intensity values of the image using the normalized cumulative histogram.
    # print(im_orig[127, 127])
    im_eq = cum_hist[im_orig]
    # print(im_eq[127, 127])
    hist_eq = np.histogram(im_eq, bins=256)[0]
    print(hist_eq)
    return [im_eq, hist_orig, hist_eq]


if __name__ == '__main__':
    grad, norm_grad = grad_img()
    # print(grad_img().ndim)
    # read_image('../huji_logo_rbg.png', 1)
    # imdisplay('../huji_logo_rbg.png', 1)
    # im = rgb2yiq(read_image('../huji_logo_rbg.png', 2))
    # plt.imshow(im[:,:,0], cmap='gray')
    # plt.show()
    # plt.imshow(yiq2rgb(im))
    # plt.show()
    # histogram_equalize(grad)
    histogram_equalize(norm_grad)
