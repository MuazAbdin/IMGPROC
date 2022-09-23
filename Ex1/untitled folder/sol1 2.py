import numpy as np
import matplotlib.pyplot as plt
import imageio

from skimage.color import rgb2gray
from typing import List

GRAYSCALE = 1
RBG = 2

RGB2YIQ_MATRIX = np.array([[0.299, 0.587, 0.114],
                           [0.596, -0.275, -0.321],
                           [0.212, -0.523, 0.311]])


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
    ax.imshow(read_image(filename, representation), cmap=cmap)
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
    return np.dot(imRGB[:, :, :3], RGB2YIQ_MATRIX.T)
    # return imRGB[:, :, :3] @ RGB2YIQ_MATRIX.T


def yiq2rgb(imYIQ):
    """
    This function transforms an image from the YIQ color space into the RGB one.
    :param imYIQ: an YIQ image.
    :type imYIQ: array_like[shape=(.., .., 3), dtype='float64']
    :return: an RBG image.
    :rtype: array_like[shape=(.., .., 3), dtype='float64']
    """
    return np.dot(imYIQ[:, :, :3], np.linalg.inv(RGB2YIQ_MATRIX).T)
    # return imYIQ[:, :, :3] @ np.linalg.inv(RGB2YIQ_MATRIX).T


#################################
## (4) Histogram equalization  ##
#################################

def histogram_equalize(im_orig: np.ndarray) -> List:
    """
    This Function performs histogram equalization of a given grayscale or RGB image.
    :param im_orig: is the input grayscale or RGB float64 image with values in [0, 1].
    :type im_orig: array_like[dtype='float64'].
    :return: a list [im_eq, hist_orig, hist_eq] where:
             im_eq - is the equalized image. grayscale or RGB float64 image with values in [0, 1].
             hist_orig - is a 256 bin histogram of the original image (array with shape (256,) ).
             hist_eq - is a 256 bin histogram of the equalized image (array with shape (256,) ).
    """
    # get the corresponding Y channel of the RGB image
    is_rgb = im_orig.ndim == 3
    if is_rgb:
        yiq = rgb2yiq(im_orig)
        im_orig = yiq[:, :, 0]

    # The image histogram.
    hist_orig, hist_orig_bins = np.histogram(im_orig, bins=256, range=(0, 1))
    # The cumulative histogram.
    cum_hist = hist_orig.cumsum()

    # The normalized cumulative histogram, stretched linearly in the range [0, 1]
    m = np.argmax(cum_hist != 0)
    cum_hist = np.around(255 * (cum_hist - cum_hist[m]) / (cum_hist[255] - cum_hist[m]))
    cum_hist[cum_hist < 0] = 0

    # Map the intensity values of the image using the normalized cumulative histogram.
    im_eq = cum_hist[np.multiply(im_orig, 255).astype(np.uint8)] / 255.0
    hist_eq, hist_eq_bins = np.histogram(im_eq, bins=256, range=(0, 1))

    if is_rgb:
        yiq[:, :, 0] = im_eq
        im_eq = yiq2rgb(yiq)

    return [im_eq, hist_orig, hist_eq]


#####################################
## (5) Optimal image quantization  ##
#####################################

def quantize(im_orig, n_quant: int, n_iter: int) -> List:
    """
    This function performs optimal quantization of a given grayscale or RGB image.
    :param im_orig: is the input grayscale or RGB float64 image with values in [0, 1].
    :type im_orig: array_like[dtype='float64'].
    :param n_quant: is the number of intensities your output im_quant image should have.
    :param n_iter: is the maximum number of iterations of the optimization procedure.
    :return: a list [im_quant, error] where:
             im_quant - is the quantized output image. (float64 image with values in [0, 1]).
             error - is an array with shape (n_iter,) (or less) of the total intensities error
                     for each iteration of the quantization procedure.
    """
    # get the corresponding Y channel of the RGB image
    is_rgb = im_orig.ndim == 3
    if is_rgb:
        yiq = rgb2yiq(im_orig)
        im_orig = yiq[:, :, 0]

    # set the initial division of z
    hist_orig, hist_orig_bins = np.histogram(im_orig, bins=256, range=(0, 1))
    cum_hist = hist_orig.cumsum()
    init_step = cum_hist[-1] / n_quant
    z = np.array([np.argmax(cum_hist >= i * init_step) for i in range(n_quant + 1)])

    q = np.zeros(n_quant, dtype=np.float64)
    error = list()
    # iterate between q and z
    for k in range(n_iter):
        prev_z = z.copy()
        error += [0]
        for i in range(n_quant):
            start, end = z[i], z[i + 1] + 1
            weights = hist_orig[start: end]
            q[i] = np.average(np.arange(start, end), weights=weights)
            # update error for the current q and z values
            error[k] += np.inner(((q[i] - np.arange(start, end)) ** 2), weights)

        for i in range(1, n_quant):
            z[i] = np.around((q[i - 1] + q[i]) / 2.0)

        if np.array_equal(z, prev_z):
            # error.pop()
            break

    # create the look up table
    LUT = np.zeros(256)
    for i in range(n_quant):
        LUT[z[i]: z[i + 1] + 1] = q[i]
    im_quant = LUT[np.multiply(im_orig, 255).astype(np.uint8)] / 255.0

    if is_rgb:
        yiq[:, :, 0] = im_quant
        im_quant = yiq2rgb(yiq)

    return [im_quant, error]


def quantize_rgb(im_orig, n_quant: int):
    """
    This function performs a quantization for full color images.
    :param im_orig: is the input grayscale or RGB float64 image with values in [0, 1].
    :type im_orig: array_like[dtype='float64'].
    :param n_quant: is the number of intensities your output im_quant image should have.
    :return: im_quant - is the quantized output image. (float64 image with values in [0, 1]).
    """
    def widest(color_boxes):
        univ_widest = -1
        for box in color_boxes:
            widest_channel = np.argmax([np.ptp(box[:, :, c]) for c in range(3)])

    if im_orig.ndim == 2:
        return quantize(im_orig, n_quant, 100)[0]
    boxes = np.array([im_orig])

    for i in range(n_quant):
        widest_box = np.argmax([np.ptp(im_orig[:, :, b]) for b in range(3)])
        # median_of_widest_channel =

        # img_arr = img_arr[img_arr[:, space_with_highest_range].argsort()]

    return
