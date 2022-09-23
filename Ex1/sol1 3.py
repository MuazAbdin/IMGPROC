import numpy as np
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
from scipy.misc import imread as imread

# Constants
PIXEL_INTENSITY_MAX = 255
PIXEL_INTENSITIES = PIXEL_INTENSITY_MAX + 1
PIXEL_RANGE = (0, PIXEL_INTENSITIES)
PIXEL_RANGE_NORMALIZED = (0, 1)
PIXEL_CHANNELS_RGB = 3
PIXEL_CHANNELS_RGBA = 4

# Picture representation modes
MODE_GRAYSCALE = 1
MODE_RGB = 2

# Magic matrices
MATRIX_RGB2YIQ = np.array([[0.299, 0.587, 0.114],
                           [0.596, -0.275, -0.321],
                           [0.212, -0.523, 0.311]])
MATRIX_YIQ2RGB = np.linalg.inv(MATRIX_RGB2YIQ)

# YIQ Properties
INDEX_Y = 0

# GREYSCALE Properties
GREYSCALE_AXES = 2

# RGB Properties
RGB_AXES = 3


# Methods
def grad_img():
    x = np.hstack([np.repeat(np.arange(0, 50, 2), 10)[None, :], np.array([255] * 6)[None, :]])
    grad = np.tile(x, (256, 1))
    norm_grad = grad.astype(np.float64) / grad.max()
    # print(np.histogram(grad, bins=256, range=(0, 255))[0])
    # plt.imshow(grad, cmap='gray')
    # plt.savefig('grad_gray.png')
    # plt.show()
    # plt.imshow(grad)
    # plt.savefig('grad.png')
    # plt.show()
    return grad, norm_grad

def read_image(filename, representation):
    """
    reads an image file and converts it into a given representation.
    :param filename:        string containing the image filename to read.
    :param representation:  representation code, either 1 or 2 defining whether
                            the output should be a greyscale image (1) or an
                            RGB image (2).
    :return:                returns an image represented by a matrix of type
                            .float64 with intensities normalized to the
                            range [0,1]
    """
    im = imread(filename)
    im_float = im.astype(np.float64)
    if (representation == MODE_GRAYSCALE):
        im_float = rgb2gray(im_float)
    return im_float / PIXEL_INTENSITY_MAX


def display_image(image, cmap=None):
    clipped = np.clip(image[..., :], 0, 1)
    plt.imshow(clipped, cmap=cmap)
    plt.show()


def imdisplay(filename, representation):
    """
    reads an image file and displays the loaded image in converted
    representation.
    figure.
    :param filename:        string containing the image filename to read.
    :param representation:  representation code, either 1 or 2 defining whether
                            the output should be a greyscale image (1) or an
                            RGB
                            image (2).
    """
    im = read_image(filename, representation)
    cmap = None
    if (representation == MODE_GRAYSCALE):
        cmap = plt.cm.gray
    display_image(im, cmap)


def is_greyscale(image):
    """
    Returns true if image is greyscale, otherwise false.
    :param image:   an image to check
    :return:        true if image is greyscale, otherwise false
    """
    # Image is just (width * height) with no pixel axis
    return (len(image.shape) == GREYSCALE_AXES)


def rgb2yiq(imRGB):
    """
    Transforms an RGB image into YIQ color space.
    Removes alpha channel if exists.
    :param imRGB:   an RGB image.
    :return:        an YIQ image with the same dimensions as the input.
    """
    return np.dot(imRGB[:, :, :PIXEL_CHANNELS_RGB], MATRIX_RGB2YIQ.T)


def yiq2rgb(imYIQ):
    """
    Transforms an RGB image into YIQ color space.
    Removes alpha channel if exists.
    :param imRGB:   an RGB image.
    :return:        an YIQ image with the same dimensions as the input.
    """
    return np.dot(imYIQ[:, :, :PIXEL_CHANNELS_RGB], MATRIX_YIQ2RGB.T)


def intensity_histogram_translated(intensities, translate=True):
    """
    Returns a histogram of the given intensities translated to [0, 255] range.
    Translates the intensities from [0,1] to [0, 255] if translate=True.
    :param intensities: An array of intensities, in [0, 1] range by default.
    :param translate:   Flag if the intensities are given in [0, 1] range.
                        Should be False if the given intensities are already
                        in [0, 255] range.
    :return:    A histogram of the intensities, in [0, 255] range.
    """
    # Translate [0,1] intensity range to [0,255] integer range
    if (translate):
        intensities = np.round(intensities * PIXEL_INTENSITY_MAX).astype(
            np.uint8)

    # Build cumulative histogram of pixel intensities
    hist, bins = np.histogram(a=intensities, bins=PIXEL_INTENSITIES,
                              range=PIXEL_RANGE)

    # Don't give a damn about bins, return hist
    return hist


def intensity_equalize(intensities):
    """
    Performs histogram equalization of the given pixel intensities.
    :param intensities: float64 image intensities with values in [0, 1].
    :return:            list [im_eq, hist_orig, hist_eq] where:
                        intensity_eq - is the equalized intensities. np.array
                        with values in [0, 1].
                        hist_orig - is a 256 bin histogram of the original
                        intensities (array with shape (256,) ).
                        hist_eq - is a 256 bin histogram of the
                        equalized intensities (array with shape (256,) ).
    """
    # Translate [0,1] intensity range to [0,255] integer range
    translated_intensities = np.round(intensities * PIXEL_INTENSITY_MAX).astype(
        np.uint8)

    # Build cumulative histogram of pixel intensities
    hist_orig = intensity_histogram_translated(translated_intensities, False)
    cdf = np.cumsum(hist_orig)

    # Normalize cumulative histogram:
    # C[k] = round(((C[k] / NUM_OF_PIXELS) * PIXEL_INTENSITY_MAX))
    cdf_n = (cdf * PIXEL_INTENSITY_MAX / cdf[-1]).astype(np.uint8)

    # Map original intensity values to their equalized values
    intensity_eq = np.array(list(map(lambda i: cdf_n[i],
                                     translated_intensities))).astype(np.uint8)

    # Calculate histogram of equalized intensity values
    hist_eq, bins = np.histogram(intensity_eq, PIXEL_INTENSITIES)

    return (intensity_eq / PIXEL_INTENSITY_MAX), hist_orig, hist_eq


def histogram_equalize(im_orig):
    """
    Performs histogram equalization of a given grayscale or RGB image.
    :param im_orig: Grayscale or RGB float64 image with values in [0, 1].
    :return:        list [im_eq, hist_orig, hist_eq] where:
                    im_eq - is the equalized image. grayscale or RGB float64
                    image with values in [0, 1].
                    hist_orig - is a 256 bin histogram of the original image
                    (array with shape (256,) ).
                    hist_eq - is a 256 bin histogram of the equalized image
                    (array with shape (256,) ).
    """
    # GREYSCALE Image
    if (is_greyscale(im_orig)):
        # Equalize image intensities
        intensities_eq, hist_orig, hist_eq = intensity_equalize(im_orig)
        im_eq = intensities_eq

    # RGB Image
    else:
        # Convert RGB [0-1] to YIQ [0-255] for equalizing Y values
        yiq = rgb2yiq(im_orig)
        y_eq, hist_orig, hist_eq = intensity_equalize(yiq[..., INDEX_Y])

        # Update Y with equalized values and go back to RGB
        yiq[..., INDEX_Y] = y_eq
        im_eq = yiq2rgb(yiq)

    return [im_eq, hist_orig, hist_eq]


def quantize(im_orig, n_quant, n_iter):
    """
    Performs optimal quantization of a given greyscale or RGB image.
    :param im_orig: input grayscale or RGB image to be quantized (float64
                    image with values in [0, 1])
    :param n_quant: number of intensities the output im_quant image should
                    have.
    :param n_iter:  maximum number of iterations of the optimization
                    procedure (may converge earlier.)
    :return:        list [im_quant, error] where:
                    im_quant - is the quantized output image.
                    error - is an array with shape (n_iter,) (or less) of
                    the total intensities error for each iteration of the
                    quantization procedure.
    """
    # Build histogram of pixel intensities
    if (is_greyscale(im_orig)):
        hist_orig = intensity_histogram_translated(im_orig)
    else:
        yiq = rgb2yiq(im_orig)
        hist_orig = intensity_histogram_translated(yiq[..., INDEX_Y])

    # Distribute pixels ranges by equal cumulative sum
    z_arr = np.arange(n_quant + 1)
    cdf = hist_orig.cumsum()
    z_space = cdf[-1] / n_quant
    z_cumsums = np.linspace(z_space, cdf[-1], n_quant)
    for i in range(n_quant):
        z_arr[i + 1] = np.argmin(cdf < z_cumsums[i])

    # Initial guess: q are medians of each range
    q_arr = np.arange(n_quant)
    for i in range(n_quant):
        start, end = z_arr[i], z_arr[i + 1] + 1
        q_arr[i] = (start + end) / 2

    # Initialize errors array
    error = np.array([0] * n_iter)

    print(f'{z_arr}  z.shape = {z_arr.shape}')
    print(f'{q_arr}  z.shape = {q_arr.shape}')

    # Iterate until n_iter exceeded or z_arr did not change
    for j in range(n_iter):
        print(f'>> k={j}')
        # Reset iteration error
        error_j = 0

        # Store previous z values to check for convergence
        z_arr_prev = z_arr.copy()

        # Calculate q values for current z
        for i in range(len(z_arr) - 1):
            start, end = z_arr[i], z_arr[i + 1] + 1
            szp = sum(hist_orig[start:end] * np.arange(start, end))
            sp = sum(hist_orig[start:end])
            q_arr[i] = szp / sp
            error_j += np.sum((hist_orig[start:end] - q_arr[i]) ** 2)

        # Calculate z values by updated q
        for i in range(1, len(q_arr)):
            z_arr[i] = (q_arr[i - 1] + q_arr[i]) / 2

        print(f'q={q_arr}')
        print(f'error={error}')
        print(f'z={z_arr}')

        # Stop iterating upon convergence
        if (np.array_equal(z_arr, z_arr_prev)):
            # Yay! we have converged! :-D
            break

        # Record iteration error
        error[j] = error_j

    # Build quantization lookup table
    lut = np.arange(PIXEL_INTENSITIES)
    for i in range(len(z_arr) - 1):
        start, end = z_arr[i], z_arr[i + 1]
        lut[start:end + 1] = q_arr[i]

    if (is_greyscale(im_orig)):
        intensities = im_orig
    else:
        yiq = rgb2yiq(im_orig)
        intensities = yiq[..., INDEX_Y]

    # Translate [0,1] intensity range to [0,255] integer range
    intensities = np.round(intensities * PIXEL_INTENSITY_MAX).astype(
        np.uint8)

    # Map intensity values to their quantized values
    intensities_quant = np.array(list(map(lambda i: lut[i],
                                          intensities))).astype(np.uint8)
    # Translate [0,255] intensity range back to [0,1]
    intensities_quant = intensities_quant / PIXEL_INTENSITIES

    if (is_greyscale(im_orig)):
        im_quant = intensities_quant
    else:
        yiq[..., INDEX_Y] = intensities_quant
        im_quant = yiq2rgb(yiq)

    # Woohoo!
    print(im_quant)
    return im_quant, error

# # -*- coding: utf-8 -*-
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.misc import imread as imread, imsave as imsave
# from skimage.color import rgb2gray
#
# def grad_img():
#     x = np.hstack([np.repeat(np.arange(0, 50, 2), 10)[None, :], np.array([255] * 6)[None, :]])
#     grad = np.tile(x, (256, 1))
#     norm_grad = grad.astype(np.float64) / grad.max()
#     # print(np.histogram(grad, bins=256, range=(0, 255))[0])
#     # plt.imshow(grad, cmap='gray')
#     # plt.savefig('grad_gray.png')
#     # plt.show()
#     # plt.imshow(grad)
#     # plt.savefig('grad.png')
#     # plt.show()
#     return grad, norm_grad
#
# def read_image(filename, representation):
#     """
#     Read image
#     @filename: file name
#     @representation: 1 == gray, other=RGB
#     """
#     im = imread(filename)
#     if representation == 1:
#         return rgb2gray(im).astype(np.float32)
#     im_float = im.astype(np.float32)
#     im_float /= 255
#     return im_float
#
#
# def imdisplay(im, representation):
#     """
#     Display image
#     @im: image to display
#     @representation: 1==gray, other=RGB
#     """
#     if representation == 1:
#         plt.imshow(im, cmap=plt.cm.gray)
#     else:
#         plt.imshow(im)
#
#
# def getMatrix():
#     """
#     Return the matrix to convert from RGB to YIQ
#     """
#     return np.array([[0.299, 0.587, 0.114],
#                      [0.596, -0.275, -0.321],
#                      [0.212, -0.523, 0.311]])
#
#
# def mul_matrix(a):
#     """
#     Mul matrix
#     """
#     return np.dot(getMatrix(), np.array(a))
#
#
# def rgb2yiq(imRGB):
#     """
#     convert from RGB to YIQ
#     """
#     return np.apply_along_axis(mul_matrix, 2, imRGB)
#
#
# def mul_rev_matrix(a):
#     """
#     mul the invert matrix
#     """
#     return np.dot(np.linalg.inv(getMatrix()), a)
#
#
# def yiq2rgb(imYIQ):
#     """
#     convert from YIQ to RGB
#     """
#     return np.apply_along_axis(mul_rev_matrix, 2, imYIQ)
#
#
# def histogram_equalize(im_orig):
#     """
#     Preform a histogram equlization
#     """
#     cut_im = im_orig
#     ret = im_orig
#     if len(im_orig.shape) > 2:  # RGB image
#         ret = rgb2yiq(im_orig)
#         cut_im = ret[:, :, 0]
#     im256 = (cut_im * 255).round().astype(np.uint8)
#     hist_orig, bins = np.histogram(im256, 256, [0, 256])
#     chist = np.cumsum(hist_orig).astype(np.float32)
#     chist = chist / chist[-1] * 255
#     cdf_m = np.ma.masked_equal(chist, 0)
#     cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
#     streach = np.ma.filled(cdf_m, 0).astype('uint8')
#     streach = np.around(streach)
#     temp = np.interp(im256.flatten(), bins[:-1], streach, right=1, left=0)
#     temp = temp / 255
#     if len(im_orig.shape) > 2:  # RGB image
#         ret[:, :, 0] = temp.reshape(cut_im.shape)
#         im_eq = np.clip(yiq2rgb(ret), 0, 1)
#         calc_hist_eq = ret[:, :, 0]
#     else:
#         im_eq = temp.reshape(cut_im.shape)
#         calc_hist_eq = im_eq
#
#     return (im_eq, hist_orig, np.histogram((calc_hist_eq * 255).astype(np.uint8),
#                                            256, [0, 256])[0])
#
#
# def find_nearest(array, value):
#     """
#     find the index in array that is nearest to value
#     """
#     idx = (np.abs(array - value)).argmin()
#     return idx
#
#
# def quantize(im_orig, n_quant, n_iter):
#     """
#     preform quatize
#     """
#     cut_im = im_orig
#     ret = im_orig
#     if len(im_orig.shape) > 2:  # RGB image
#         ret = rgb2yiq(im_orig)
#         cut_im = ret[:, :, 0]
#     im256 = (cut_im * 255).round().astype(np.uint8)
#     hist_orig, bins = np.histogram(im256, 256, [0, 256])
#     z = np.zeros(n_quant + 1, dtype=np.uint8)
#     sum_hist = np.cumsum(hist_orig)
#     for i in range(1, n_quant):
#         z[i] = find_nearest(np.ma.masked_equal(sum_hist, 0),
#                             i / n_quant * sum_hist[-1])
#
#     z[n_quant] = 255
#     print(f'{z}  z.shape = {z.shape}')
#     error = list()
#     q = np.zeros(n_quant, dtype=np.float32)
#     print(f'{q}  z.shape = {q.shape}')
#     for iteration in range(n_iter):
#         print(f'>>> k={iteration}')
#         last_z = np.ndarray.copy(z)
#         for i in range(len(q)):
#             sumD = np.sum(hist_orig[z[i]: z[i + 1] + 1])
#             temp = np.arange(z[i], z[i + 1] + 1)
#             sumM = np.sum(np.multiply(hist_orig[z[i]: z[i + 1] + 1], temp))
#             q[i] = sumM / sumD
#         for i in range(1, len(z) - 1):
#             z[i] = round((q[i - 1] + q[i]) / 2.0)
#         error_sum = 0
#         for i in range(n_quant):
#             temp = np.arange(z[i], z[i + 1])
#             temp = np.apply_along_axis(lambda x: (q[i] - x) ** 2, 0, temp)
#             error_sum += np.sum(np.multiply(hist_orig[z[i]:z[i + 1]], temp))
#         error.append((error_sum))
#
#         print(f'q={q}')
#         print(f'error={error}')
#         print(f'z={z}')
#
#         if (np.asarray(last_z) == np.asarray(z)).all():
#             break
#
#     temp = cut_im.flatten()
#     im256 = im256.flatten()
#     # Build LUT
#     lut = np.array([q[0]], dtype=np.uint8)
#     for i in range(n_quant):
#         lut = np.append(lut, np.array([q[i]] * (z[i + 1] - z[i])))
#     temp = lut[im256]
#     temp = temp / 255
#     if len(im_orig.shape) > 2:  # RGB image
#         ret[:, :, 0] = temp.reshape(cut_im.shape)
#         im_quant = yiq2rgb(ret)
#         # im_quant = np.clip(im_quant, 0, 1)
#     else:
#         im_quant = temp.reshape(cut_im.shape)
#
#     print(im_quant)
#
#     return (im_quant, error)


# import numpy as np
# import matplotlib.pyplot as plt
# import imageio
#
# from skimage.color import rgb2gray
# from typing import List
#
# GRAYSCALE = 1
# RBG = 2
#
# RGB2YIQ_MATRIX = np.array([[0.299, 0.587, 0.114],
#                            [0.596, -0.275, -0.321],
#                            [0.212, -0.523, 0.311]])
#
#
# def grad_img():
#     x = np.hstack([np.repeat(np.arange(0, 50, 2), 10)[None, :], np.array([255] * 6)[None, :]])
#     grad = np.tile(x, (256, 1))
#     norm_grad = grad.astype(np.float64) / grad.max()
#     # print(np.histogram(grad, bins=256, range=(0, 255))[0])
#     # plt.imshow(grad, cmap='gray')
#     # plt.savefig('grad_gray.png')
#     # plt.show()
#     # plt.imshow(grad)
#     # plt.savefig('grad.png')
#     # plt.show()
#     return grad, norm_grad
#
#
# #######################################################
# ## (1) Reading an image into a given representation  ##
# #######################################################
#
# def read_image(filename: str, representation: int):
#     """
#     This function reads an image into a given representation.
#     :param filename: the filename of an image on disk (could be grayscale or RGB).
#     :param representation: representation code, either 1 or 2 defining whether the
#                            output should be a grayscale image (1) or an RGB image (2).
#     :return: an image represented by a matrix of type np.float64 with
#              intensities normalized to the range [0, 1].
#     :rtype: array_like['float64']
#     """
#     # make sure the matrix type is np.float64.
#     img = imageio.imread(filename).astype(np.float64)
#     # normalize intensities to the range [0, 1].
#     img /= img.max()
#     return img if representation == RBG else rgb2gray(img)
#
#
# ##############################
# ## (2) Displaying an image  ##
# ##############################
#
# def imdisplay(filename: str, representation: int) -> None:
#     """
#     This function opens a new figure and display the loaded image in the converted
#     representation.
#     :param filename: the filename of an image on disk.
#     :param representation: representation code.
#     :return: nothing.
#     """
#     cmap = None if representation == RBG else 'gray'
#     fig, ax = plt.subplots()
#     img = ax.imshow(read_image(filename, representation), cmap=cmap)
#
#     plt.colorbar(img, orientation='horizontal')
#     fig.tight_layout()
#     fig.show()
#
#
# #######################################################
# ## (3) Transforming an RGB image to YIQ color space  ##
# #######################################################
#
# def rgb2yiq(imRGB):
#     """
#     This function transforms an RGB image into the YIQ color space.
#     :param imRGB: an RBG image.
#     :type imRGB: array_like[shape=(.., .., 3), dtype='float64']
#     :return: an YIQ image.
#     :rtype: array_like[shape=(.., .., 3), dtype='float64']
#     """
#     # return np.dot(imRGB[:, :, :3], RGB2YIQ_MATRIX.T)
#     return imRGB[:, :, :3] @ RGB2YIQ_MATRIX.T
#
#
# def yiq2rgb(imYIQ):
#     """
#     This function transforms an image from the YIQ color space into the RGB one.
#     :param imYIQ: an YIQ image.
#     :type imYIQ: array_like[shape=(.., .., 3), dtype='float64']
#     :return: an RBG image.
#     :rtype: array_like[shape=(.., .., 3), dtype='float64']
#     """
#     # return np.dot(imYIQ[:, :, :3], np.linalg.inv(RGB2YIQ_MATRIX).T)
#     return imYIQ[:, :, :3] @ np.linalg.inv(RGB2YIQ_MATRIX).T
#
#
# #################################
# ## (4) Histogram equalization  ##
# #################################
#
# def histogram_equalize(im_orig: np.ndarray) -> List:
#     """
#     This Function performs histogram equalization of a given grayscale or RGB image.
#     :param im_orig: is the input grayscale or RGB float64 image with values in [0, 1].
#     :type im_orig: array_like[dtype='float64'].
#     :return: a list [im_eq, hist_orig, hist_eq] where:
#              im_eq - is the equalized image. grayscale or RGB float64 image with values in [0, 1].
#              hist_orig - is a 256 bin histogram of the original image (array with shape (256,) ).
#              hist_eq - is a 256 bin histogram of the equalized image (array with shape (256,) ).
#     """
#     # get the corresponding Y channel of the RGB image
#     is_rgb = im_orig.ndim == 3
#     if is_rgb:
#         yiq = rgb2yiq(im_orig)
#         im_orig = yiq[:, :, 0]
#
#     # The image histogram.
#     hist_orig, hist_orig_bins = np.histogram(im_orig, bins=256, range=(0, 1))
#     assert hist_orig.shape == (256,)
#
#     # The cumulative histogram.
#     cum_hist = hist_orig.cumsum()
#
#     # The normalized cumulative histogram, stretched linearly in the range [0, 1]
#     m = np.argmax(cum_hist != 0)
#     cum_hist = np.around(255 * (cum_hist - cum_hist[m]) / (cum_hist[255] - cum_hist[m]))
#     cum_hist[cum_hist < 0] = 0
#
#     # Map the intensity values of the image using the normalized cumulative histogram.
#     im_eq = cum_hist[np.multiply(im_orig, 255).astype(np.uint8)] / 255.0
#     assert 0 <= np.min(im_eq) and np.max(im_eq) <= 1 and im_eq.dtype == np.float64
#     hist_eq, hist_eq_bins = np.histogram(im_eq, bins=256, range=(0, 1))
#     assert hist_eq.shape == (256,)
#
#     if is_rgb:
#         yiq[:, :, 0] = im_eq
#         im_eq = yiq2rgb(yiq)
#
#     return [im_eq, hist_orig, hist_eq]
#
#
# #####################################
# ## (5) Optimal image quantization  ##
# #####################################
#
# def quantize(im_orig, n_quant: int, n_iter: int) -> List:
#     """
#     This function performs optimal quantization of a given grayscale or RGB image.
#     :param im_orig: is the input grayscale or RGB float64 image with values in [0, 1].
#     :type im_orig: array_like[dtype='float64'].
#     :param n_quant: is the number of intensities your output im_quant image should have.
#     :param n_iter: is the maximum number of iterations of the optimization procedure.
#     :return: a list [im_quant, error] where:
#              im_quant - is the quantized output image. (float64 image with values in [0, 1]).
#              error - is an array with shape (n_iter,) (or less) of the total intensities error
#                      for each iteration of the quantization procedure.
#     """
#     # get the corresponding Y channel of the RGB image
#     is_rgb = im_orig.ndim == 3
#     if is_rgb:
#         yiq = rgb2yiq(im_orig)
#         im_orig = yiq[:, :, 0]
#
#     # set the initial division of z
#     hist_orig, hist_orig_bins = np.histogram(im_orig, bins=256, range=(0, 1))
#     cum_hist = hist_orig.cumsum()
#     init_step = cum_hist[-1] / n_quant
#     z = np.array([np.argmax(cum_hist >= i * init_step) for i in range(n_quant + 1)])
#     print(f'{z}  z.shape = {z.shape}')
#
#     q = np.zeros(n_quant, dtype=np.float64)
#     print(f'{q}  q.shape = {q.shape}')
#     error = list()
#     for k in range(n_iter):
#         print(f'>>> k={k}')
#         prev_z = z.copy()
#         error += [0]
#         for i in range(n_quant):
#             start, end = z[i], z[i+1] + 1
#             # print(f'start={start}  end={end}')
#             weights = hist_orig[start: end]
#             # print(weights)
#             # q[i] = np.inner(weights, np.arange(start, end)) / sum(weights)
#             q[i] = np.average(np.arange(start, end), weights=weights)
#             # print(f'{np.average(np.arange(start, end), weights=weights)} '
#             #       f'{np.inner(weights, np.arange(start, end)) / sum(weights)} '
#             #       f'{sum(weights * np.arange(start, end)) / sum(weights)}')
#             # q[i] = sum(weights * np.arange(start, end)) / sum(weights)
#             # print(f'q[{i}]={q[i]}')
#             # print((q[i] - np.arange(start, end)) ** 2)
#             # error[k] += sum(((q[i] - np.arange(start, end)) ** 2) * weights)
#             error[k] += np.inner(((q[i] - np.arange(start, end)) ** 2), weights)
#             # print(f'error[{k}]={error[k]}')
#
#         print(f'q={q}')
#         print(f'error={error}')
#
#         for i in range(1, n_quant):
#             z[i] = (q[i-1] + q[i]) / 2
#         print(f'z={z}')
#
#         if np.array_equal(z, prev_z):
#             break
#
#     LUT = np.zeros(256)
#     for i in range(n_quant):
#         # LUT[z[i] <= hist_orig < z[i+1]] = q[i]
#         LUT[z[i]: z[i + 1] + 1] = q[i]
#     # print(LUT)
#     im_quant = LUT[np.multiply(im_orig, 255).astype(np.uint8)] / 255.0
#     print(im_quant)
#
#     if is_rgb:
#         yiq[:, :, 0] = im_quant
#         im_quant = yiq2rgb(yiq)
#
#     return [im_quant, error]
#     # return []


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

    # plt.imshow(norm_grad, cmap='gray')
    # plt.show()
    # plt.imshow(histogram_equalize(norm_grad)[0], cmap='gray')
    # plt.show()
    # histogram_equalize(norm_grad)

    # a = np.array([7, 8, 1, 2, 3, 1, 5, 1, 6, 1])
    # print(f'{a == 1} {np.argmax(a == 1)}   {np.amin(a == 1)}')

    # temp = np.hstack([np.repeat(np.arange(50, 100, 2), 10)[None, :], np.array([255] * 6)[None, :]])
    # grad_im = np.tile(temp, (256, 1))
    # norm_grad_im = grad_im.astype(np.float64) / grad_im.max()
    # histogram_equalize(norm_grad_im)
    # print(np.histogram(grad_im, bins=255, range=(0, 255))[0])
    # plt.imshow(grad_im, cmap='gray')
    # plt.show()

    # n_quant = 16
    # hist_orig, hist_orig_bins = np.histogram(norm_grad, bins=256, range=(0, 1))
    # # print(hist_orig)
    # cum_hist = hist_orig.cumsum()
    # # print(cum_hist)
    # init_step = cum_hist[-1] / n_quant
    # # print(init_step)
    # z = np.array([np.argmax(cum_hist >= i * init_step) for i in range(n_quant + 1)])
    # # print(z)
    # # for i in range(n_quant + 1):
    # #     z[i] = np.argmax(cum_hist >= i * init_step)
    # new_his = np.histogram(grad, bins=z, range=(0, 255))[0]
    # print(new_his)
    # plt.bar(z[:-1], new_his, width=1)
    # plt.show()

    # z_arr = np.arange(32 + 1)
    # cdf = hist_orig.cumsum()
    # z_space = cdf[-1] / 32
    # z_cumsums = np.linspace(z_space, cdf[-1], 32)
    # print(z_cumsums)
    # for i in range(32):
    #     z_arr[i + 1] = np.argmin(cdf < z_cumsums[i])
    # print(z_arr)

    res = quantize(histogram_equalize(norm_grad)[0], 5, 25)
    print(res[1])
    plt.imshow(res[0], cmap='gray')
    plt.show()
    # quantize(norm_grad, 16, 25)
    # [14641145.05847958, 14386611.36202407, 14133100.246572616, 13800172.246573092, 13800172.246573092]
    # [22970971.42857143, 16740443.42857143, 14742875.42857143, 14742875.42857143]