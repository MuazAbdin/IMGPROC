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
    assert hist_orig.shape == (256,)

    # The cumulative histogram.
    cum_hist = hist_orig.cumsum()

    # The normalized cumulative histogram, stretched linearly in the range [0, 1]
    m = np.argmax(cum_hist != 0)
    cum_hist = np.around(255 * (cum_hist - cum_hist[m]) / (cum_hist[255] - cum_hist[m]))
    cum_hist[cum_hist < 0] = 0

    # Map the intensity values of the image using the normalized cumulative histogram.
    im_eq = cum_hist[np.multiply(im_orig, 255).astype(np.uint8)] / 255.0
    assert 0 <= np.min(im_eq) and np.max(im_eq) <= 1 and im_eq.dtype == np.float64
    hist_eq, hist_eq_bins = np.histogram(im_eq, bins=256, range=(0, 1))
    assert hist_eq.shape == (256,)

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
    print(f'{z}  z.shape = {z.shape}')

    q = np.zeros(n_quant, dtype=np.float64)
    print(f'{q}  q.shape = {q.shape}')
    error = list()
    for k in range(n_iter):
        print(f'>>> k={k}')
        prev_z = z.copy()
        error += [0]
        for i in range(n_quant):
            start, end = z[i], z[i+1] + 1
            # print(f'start={start}  end={end}')
            weights = hist_orig[start: end]
            # print(weights)
            # q[i] = np.inner(weights, np.arange(start, end)) / sum(weights)
            q[i] = np.average(np.arange(start, end), weights=weights)
            # print(f'{np.average(np.arange(start, end), weights=weights)} '
            #       f'{np.inner(weights, np.arange(start, end)) / sum(weights)} '
            #       f'{sum(weights * np.arange(start, end)) / sum(weights)}')
            # q[i] = sum(weights * np.arange(start, end)) / sum(weights)
            # print(f'q[{i}]={q[i]}')
            # print((q[i] - np.arange(start, end)) ** 2)
            # error[k] += sum(((q[i] - np.arange(start, end)) ** 2) * weights)
            error[k] += np.inner(((q[i] - np.arange(start, end)) ** 2), weights)
            # print(f'error[{k}]={error[k]}')

        print(f'q={q}')
        print(f'error={error}')

        for i in range(1, n_quant):
            z[i] = (q[i-1] + q[i]) / 2
        print(f'z={z}')

        if np.array_equal(z, prev_z):
            break

    LUT = np.zeros(256)
    for i in range(n_quant):
        # LUT[z[i] <= hist_orig < z[i+1]] = q[i]
        LUT[z[i]: z[i + 1] + 1] = q[i]
    # print(LUT)
    im_quant = LUT[np.multiply(im_orig, 255).astype(np.uint8)] / 255.0
    print(im_quant)

    if is_rgb:
        yiq[:, :, 0] = im_quant
        im_quant = yiq2rgb(yiq)

    return [im_quant, error]
    # return []


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

    plt.imshow(quantize(histogram_equalize(norm_grad)[0], 5, 25)[0], cmap='gray')
    plt.show()
    # quantize(norm_grad, 16, 25)