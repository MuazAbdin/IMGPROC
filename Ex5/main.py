import time
from typing import Dict

import imageio
import numpy as np
import matplotlib.pyplot as plt
from algorithms import *
from skimage.color import rgb2gray

DEFAULT_DOMAIN = np.linspace(-25, 25)
SORTING_ALGORITHMS = {'Bubble': bubble_sort, 'Selection': selection_sort,
                      'Insertion': insertion_sort, 'Heap': heap_sort,
                      'Merge': merge_sort, 'Quick': quick_sort}
N = 50
FONT = {'family': 'Georgia',
        'color': '#FF5C5C',
        'weight': 'semibold',
        'size': 16, }

FONT1 = {'family': 'Georgia',
         'color': '#975679',
         'weight': 'semibold',
         'size': 14, }

FONT2 = {'family': 'Georgia',
         'weight': 'semibold',
         'size': 10, }

LINEPROPS = {'color': 'r',
             'linestyle': '-.',
             'linewidth': 0.5}


def run_program(array=None, num=N):
    if not array:
        array = np.random.randint(low=-100, high=100, size=15)
    results = dict()
    for name, method in SORTING_ALGORITHMS.items():
        start = time.time()
        for i in range(num):
            method(array.copy())
        end = time.time()
        results[name] = round(end - start, 5) * 1000
    return results


def plot_func(func=lambda x: x, domain=DEFAULT_DOMAIN, x_label='x', y_label='y',
              title='Identity Function', func_label='Identity', fmt='b-', save=False):
    plt.plot(domain, func(domain), fmt, label=func_label)
    plt.xlabel(x_label, fontname='Georgia')
    plt.ylabel(y_label, fontname='Georgia')
    plt.title(title)

    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(f'{title}.png')


def plot_bars():
    x = np.linspace(-100, 100, 200)
    y = np.power(x, 2)
    plt.bar(x, y)
    plt.show()


# def plot_results():
#     results = run_program(num=1000)
#     algorithms = list(results.keys())
#     runtime = list(results.values())
#     plt.bar(algorithms, runtime, color='maroon', width=0.6)
#     plt.xticks(fontname='Georgia')
#     plt.yticks(fontname='Georgia')
#     plt.xlabel("Sorting Algorithms", fontdict=FONT)
#     plt.ylabel("Runtime (MS)", fontdict=FONT)
#     plt.title("Sorting Algorithms Runtime", fontdict=FONT)
#     plt.tight_layout()
#     plt.show()


def plot_results(results: Dict[str, float]):
    algorithms, runtime = list(results.keys()), list(results.values())
    fig, ax = plt.subplots()
    fig.set_facecolor('#CFFEE0')
    ax.set_facecolor('#FFE8AA')

    ax.bar(algorithms, runtime, color='#4A6CA0', width=0.6, zorder=2)
    ax.set_title('Sorting Algorithms Runtime', pad=20, fontdict=FONT)

    ax.set_xlabel("Sorting Algorithms", fontdict=FONT1)
    ax.set_xticks(np.arange(len(algorithms)))
    ax.set_xticklabels(algorithms, fontdict=FONT2, rotation=45)

    ax.set_ylabel("Runtime (MS)", fontdict=FONT1)
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(ax.get_yticks(), fontdict=FONT2)
    # ax.yaxis.set_major_formatter('{x:1.0f}')
    ax.yaxis.set_major_formatter(format_time)
    ax.yaxis.grid(**LINEPROPS)

    fig.tight_layout()
    fig.show()


def format_time(x, pos=None):
    """The two args are the value and tick position."""
    return f'{x:1.0f}'


def read_image(filename: str, representation: int):
    # make sure the matrix type is np.float64.
    img = imageio.imread(filename).astype(np.float64)
    # normalize intensities to the range [0, 1].
    img /= img.max()
    return img if representation == 2 else rgb2gray(img).astype(np.float64)


def display_result(images_set):
    orig, im1, im2, im3, im4, final = images_set
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(20, 20))
    fig.set_facecolor('#CFFEE0')

    ax1.imshow(orig, aspect='auto')
    ax1.set_title('Original image', pad=20, fontdict=FONT)
    ax1.axis('off')

    ax3.imshow(im1, aspect='auto')
    ax3.set_title('Initial z image', pad=20, fontdict=FONT)
    ax3.axis('off')

    ax5.imshow(im2, aspect='auto')
    ax5.set_title('After 400 steps', pad=20, fontdict=FONT)
    ax5.axis('off')

    ax6.imshow(im3, aspect='auto')
    ax6.set_title('After 800 steps', pad=20, fontdict=FONT)
    ax6.axis('off')

    ax4.imshow(im4, aspect='auto')
    ax4.set_title('After 1200 steps', pad=20, fontdict=FONT)
    ax4.axis('off')

    ax2.imshow(final, aspect='auto')
    ax2.set_title('Final image', pad=20, fontdict=FONT)
    ax2.axis('off')

    # fig.tight_layout()
    fig.show()


if __name__ == '__main__':
    # functions = [np.poly1d(np.array([1, 2, 3, 4]).astype(float)),
    #              lambda x: x ** 2, lambda x: x ** 3]
    # plt.style.use('fivethirtyeight')
    # # plot_func()
    # # plt.show()
    # for f in functions[1:]:
    #     plot_func(func=f)
    #     # plt.show()
    # plt.show()

    # plot_bars()
    # print(run_program())
    # plot_results()
    # plt.style.use('fivethirtyeight')
    # plot_results(run_program(num=250))

    # print(np.__version__)
    # print(np.show_config())

    # a = np.array([2, 2, 2])
    # print(1 - a)
    # print(np.zeros(10))
    # x = np.hstack([np.repeat(np.arange(0, 50, 2), 10)[None, :], np.array([255] * 6)[None, :]])
    # grad = np.tile(x, (256, 1))
    # print(np.sum(grad, axis=0))
    # print(np.histogram(grad, bins=256)[0])
    # plt.imshow(grad, cmap='gray')
    # plt.hist(np.sum(grad, axis=0), bins=np.arange(256))
    # plt.show()

    # print(np.mean(np.arange(4).reshape(2,2), axis=0, keepdims=True))
    # print()
    # a = np.arange(24).reshape((3, 2, 4))
    # print(a)
    # print()
    # m = np.mean(a, axis=0, keepdims=True)
    # print(m)
    # print(m.shape)

    # a = np.arange(8).reshape([2,2,2])
    # print(a)
    # print('-----')
    # # print(a.transpose())
    # # print('-----')
    # # print(a.transpose([2,1,0]))
    # # print('-----')
    # print(a.transpose([2,0,1]))

    # steps = np.arange(1000)
    # # func = lambda x: x + 5
    # loss = 2 * steps + 5
    #
    # plt.plot(steps, loss, 'b-')
    # plt.xlabel('step', fontname='Georgia')
    # plt.xticks(np.arange(0, 1001, 100))
    # plt.ylabel('loss', fontname='Georgia')
    # plt.yscale('log')
    # plt.title('The Optimization Loss', fontname='Georgia')
    # plt.tight_layout()
    # plt.show()
    # plt.savefig(f'LossPlot.png')

    # import torch
    # loss_out = torch.zeros([10], dtype=torch.float32)
    # print(loss_out.size)
    # print(len(loss_out))




