import sol1

import math
import os
import itertools as it
import warnings

import numpy as np
import matplotlib.pyplot as plt
from imageio import imread

# Show outputs in a GUI if supported (be aware that figures may be quite large)
SHOULD_SHOW = False

# Save outputs as files (will be saved under a new directory called results_{timestamp})
SHOULD_SAVE = True

# Source directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep
RESOURCE_DIR = BASE_DIR + 'not_a_tester_externals' + os.sep

# Warn if quantization error is not weakly monotonically decreasing
SHOULD_CHECK_Q_ERROR = True

# In order to change/add images see the following constants
# and the parameters of each runner function bellow

TASKS_HEQ = [
    [
        np.array(
            [[52, 55, 61, 59, 79, 61, 76, 61, ],
             [62, 59, 55, 104, 94, 85, 59, 71, ],
             [63, 65, 66, 113, 144, 104, 63, 72, ],
             [64, 70, 70, 126, 154, 109, 71, 69, ],
             [67, 73, 68, 106, 122, 88, 68, 68, ],
             [68, 79, 60, 70, 77, 66, 58, 75, ],
             [69, 85, 64, 58, 55, 61, 65, 83, ],
             [70, 87, 69, 68, 65, 73, 78, 90, ]]
        ) / 255,
        np.array(
            [[0, 12, 53, 32, 190, 53, 174, 53, ],
             [57, 32, 12, 227, 219, 202, 32, 154, ],
             [65, 85, 93, 239, 251, 227, 65, 158, ],
             [73, 146, 146, 247, 255, 235, 154, 130, ],
             [97, 166, 117, 231, 243, 210, 117, 117, ],
             [117, 190, 36, 146, 178, 93, 20, 170, ],
             [130, 202, 73, 20, 12, 53, 85, 194, ],
             [146, 206, 130, 117, 85, 166, 182, 215, ]]
        ) / 255,
    ],
    [
        np.tile(np.hstack([
            np.repeat(np.arange(0, 50, 2), 10)[None, :],
            np.array([255] * 6)[None, :]
        ]), (256, 1)).astype(np.float64) / 255,
    ],
    [

        sol1.read_image(RESOURCE_DIR + 'Unequalized_Hawkes_Bay_NZ.jpg', 1),
        sol1.read_image(RESOURCE_DIR + 'Equalized_Hawkes_Bay_NZ.jpg', 1)
    ],
    [
        sol1.read_image(RESOURCE_DIR + 'low_contrast.jpg', 1)
    ],
    [
        sol1.read_image(RESOURCE_DIR + 'low_contrast.jpg', 2)
    ],
]

TASKS_Q = [
    [
        np.tile(np.hstack([
            np.repeat([0, 10, 21, 31, 41, 52, 62, 73, 83, 93, 104, 114, 124, 135,
                       145, 155, 166, 176, 187, 197, 207, 218, 228, 238, 249], 10)[None, :],
            np.array([255] * 6)[None, :]
        ]), (256, 1)).astype(np.float64) / 255,
        [1, 2, 3, 4, 5, 8, 12, 25]
    ],
    [
        sol1.read_image(RESOURCE_DIR + 'Equalized_Hawkes_Bay_NZ.jpg', 1),
        [1, 2, 3, 4, 5, 6, 8, 30]
    ],
    [
        sol1.read_image(RESOURCE_DIR + 'jerusalem.jpg', 1),
        [1, 2, 3, 4, 5, 8, 16, 30]
    ],
    [
        imread(RESOURCE_DIR + 'jerusalem.jpg').astype(np.float64) / 255,
        [1, 2, 3, 4, 5, 8, 16, 30]
    ],
    [
        sol1.read_image(RESOURCE_DIR + 'Robert_Duncanson.jpg', 1),
        [1, 2, 3, 5, 8]
    ],
]

TASKS_Q_RGB = [
    [
        sol1.read_image(RESOURCE_DIR + 'monkey.jpg', 2),
        [1, 2, 3, 4, 5, 6, 8, 16, 30, 50, 100]
    ],
    [
        imread(RESOURCE_DIR + 'jerusalem.jpg').astype(np.float64) / 255,
        [1, 2, 3, 4, 5, 6, 8, 16, 30, 50, 200]
    ],
    [
        sol1.read_image(RESOURCE_DIR + 'Robert_Duncanson.jpg', 2),
        [1, 2, 3, 4, 5, 6, 7, 8, 16, 30, 50]
    ],
    [
        sol1.read_image(RESOURCE_DIR + 'Starry_Night.jpg', 2),
        [1, 2, 3, 4, 5, 6, 7, 8, 16, 30, 50]
    ],
]

INTENSITIES = np.arange(0, 256)


def runner_histogram_equalization(im_orig, im_expected=None):
    im_result, hist_orig, hist_eq = sol1.histogram_equalize(im_orig)

    titles = ['Original', 'Expected After HEQ', 'Result After HEQ']
    images = (im_orig, im_expected, im_result)

    items = [
        (title, im) for title, im
        in zip(titles, images)
        if im is not None
    ]

    return (len(items), *(im_orig.shape[:2]), items)


def runner_quantization(im_orig, n_quants, n_iter=50, should_check_error=SHOULD_CHECK_Q_ERROR):
    def _q_generator():
        for n_quant in n_quants:
            im_q, error = sol1.quantize(im_orig, n_quant, n_iter)

            if should_check_error and np.any(np.diff(error) > 0):
                warnings.warn('Found an increase in quantization error for Q = ' + str(n_quant)
                              + ' (performed ' + str(len(error)) + ' iterations).')

            yield im_q

    total_items = len(n_quants) + 1
    titles = it.chain(('Original',), ('Result After Q = ' + str(n_quant) for n_quant in n_quants))
    images = it.chain((im_orig,), _q_generator())

    return (total_items, *(im_orig.shape[:2]), zip(titles, images))


def runner_rgb_quantization(im_orig, n_quants):
    total_items = len(n_quants) + 1
    titles = it.chain(('Original',),
                      ('Result After RGBQ = ' + str(n_quant) for n_quant in n_quants))
    images = it.chain((im_orig,), (sol1.quantize_rgb(im_orig, n_quant) for n_quant in n_quants))

    return (total_items, *(im_orig.shape[:2]), zip(titles, images))


# TODO: check/show histograms, quantification errors and iteration count (jk, probs not)

def collect_figure(total_subplots, subplot_height, subplot_width, items,
                   should_show=SHOULD_SHOW, filename=None):
    rows = math.floor(math.sqrt(total_subplots))
    cols = math.ceil(total_subplots / rows)

    # Nothing to see here, just some random HUJI magic heuristics

    if cols > rows and 3 * subplot_width >= 4 * subplot_height:
        rows, cols = cols, rows
    elif rows > cols and 3 * subplot_width <= 4 * subplot_height:
        rows, cols = cols, rows

    width = max(6.4, cols * (subplot_width / 100) + 1)
    height = max(3, rows * (subplot_height / 100) + 1)
    fontsize = math.ceil(max(width, height))

    fig, axs = plt.subplots(rows, cols, figsize=(width, height))
    axs = axs.flatten()

    for ax, (title, im) in it.zip_longest(axs, items, fillvalue=(None, None)):
        ax.set_axis_off()

        if title is not None:
            ax.set_title(title, fontsize=fontsize)

        if callable(im):
            im(ax)
        elif im is not None:
            ax.imshow(im, cmap='gray' if im.ndim == 2 else None, vmin=0, vmax=1)

    plt.tight_layout()

    if isinstance(filename, str):
        plt.savefig(filename + '.jpg')

    if should_show:
        plt.show()


def run_task_set(task_set, resolver, should_show=SHOULD_SHOW, save_prefix=None):
    for i, args in enumerate(task_set):
        print('- Running ' + str(i) + '...')
        collect_figure(*resolver(*args),
                       should_show,
                       save_prefix + str(i) if save_prefix is not None else None)


def run_task_sets(task_sets, should_show=SHOULD_SHOW, should_save=SHOULD_SAVE):
    save_dir = None

    if should_show:
        print('Output will be shown in a GUI if supported (as configured).'
              + ' Be aware that the figures may be quite large.')
    else:
        print('Output will not be shown in a GUI (as configured).')

    if should_save:
        import time

        save_dir = BASE_DIR + 'results_' + time.strftime('%Y-%m-%d_%H-%M-%S')

        if os.path.exists(save_dir):
            raise Exception('Output directory path ' + save_dir + ' already exists.')

        print('Creating output directory "' + save_dir + '"...')
        os.mkdir(save_dir)
        print('Created output directory.')

    else:
        print('Output will not be saved (as configured).')

    for (task_set, resolver, save_prefix) in task_sets:
        print('Running: ' + save_prefix.upper())
        save_prefix = save_dir + os.sep + save_prefix if should_save else None
        run_task_set(task_set, resolver, should_show, save_prefix)


def run():
    all_task_sets = [
        (TASKS_HEQ, runner_histogram_equalization, 'heq'),
        (TASKS_Q, runner_quantization, 'q')
    ]

    if 'quantize_rgb' in dir(sol1):
        all_task_sets.append((TASKS_Q_RGB, runner_rgb_quantization, 'qrgb'))

    print('Starting...')

    run_task_sets(all_task_sets)

    print('Finished')


if __name__ == "__main__":
    run()
