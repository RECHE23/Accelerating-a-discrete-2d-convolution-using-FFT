import timeit
import itertools
import numpy as np
import matplotlib.pyplot as plt


figure_id = itertools.count(1)


def display_picture(content, title="", xlabel="", figure_size=(22.5, 15)):
    id = next(figure_id)
    fig, ax = plt.subplots(figsize=figure_size, dpi=80)
    plt.imshow(content, cmap=plt.cm.gray)  # Use plt.cm.gray instead of plt.colormaps["gray"]

    # Title
    plt.title(f"Figure {id} - {title}", fontsize=14)

    # Set label position to top
    ax.xaxis.set_label_position('top')

    # Set xlabel
    ax.set_xlabel(xlabel, fontsize=11)


def plot_computation_time_by_kernel_size(kernel_range, direct_2d_convolution, fft_based_2d_convolution, number=5, repeat=5, input_size=(1200, 800), figure_size=(15, 8)):
    id = next(figure_id)
    start, end, step = kernel_range
    H, W = input_size

    f = np.random.rand(H, W)
    f[f < .999] = 0

    result = np.empty(((end - start) // step + 1, 3, repeat))
    for i, s in enumerate(range(start, end + 1, step)):
        X, Y = np.mgrid[-1:1:s * 1j, -1:1:s * 1j]
        g = np.exp(-X ** 2 * 2 - Y ** 2 * 2) ** 2

        result[i, 0] = s
        result[i, 1] = timeit.repeat(lambda: direct_2d_convolution(f, g, stride=(1, 1)), number=number, repeat=repeat)
        result[i, 1] /= number
        result[i, 1] *= 1000
        result[i, 2] = timeit.repeat(lambda: fft_based_2d_convolution(f, g, stride=(1, 1)), number=number, repeat=repeat)
        result[i, 2] /= number
        result[i, 2] *= 1000

    plt.figure(id, figsize=figure_size, dpi=80)
    plt.clf()
    plt.xticks(range(start, end + 1, step))
    plt.plot(result[:, 0, 0], np.mean(result[:, 1], axis=-1), color='b', label='Direct convolution')
    plt.fill_between(result[:, 0, 0], np.min(result[:, 1], axis=-1), np.max(result[:, 1], axis=-1), alpha=0.2,
                     edgecolor='b', facecolor='b', linewidth=1, antialiased=True)
    plt.plot(result[:, 0, 0], np.mean(result[:, 2], axis=-1), color='r', label='FFT based convolution')
    plt.fill_between(result[:, 0, 0], np.min(result[:, 2], axis=-1), np.max(result[:, 2], axis=-1), alpha=0.2,
                     edgecolor='r', facecolor='r', linewidth=1, antialiased=True)
    title = f'Computation time for a {H}px by {W}px input signal in function of kernel size'
    plt.title(f"Figure {id} - {title}", fontsize=14)
    plt.ylabel('Computation time (ms)')
    plt.xlabel('Kernel size (px) square')
    plt.legend()


def plot_computation_time_by_input_size(input_range, direct_2d_convolution, fft_based_2d_convolution, number=5, repeat=5, kernel_size=(5, 5), figure_size=(15, 8)):
    id = next(figure_id)
    start, end, step = input_range
    M, N = kernel_size

    X, Y = np.mgrid[-1:1:M * 1j, -1:1: N * 1j]
    g = np.exp(-X ** 2 * 2 - Y ** 2 * 2) ** 2

    result = np.empty(((end - start) // step + 1, 3, repeat))
    for i, s in enumerate(range(start, end + 1, step)):
        f = np.random.rand(s, s)
        f[f < .999] = 0

        result[i, 0] = s
        result[i, 1] = timeit.repeat(lambda: direct_2d_convolution(f, g, stride=(1, 1)), number=number, repeat=repeat)
        result[i, 1] /= number
        result[i, 1] *= 1000
        result[i, 2] = timeit.repeat(lambda: fft_based_2d_convolution(f, g, stride=(1, 1)), number=number, repeat=repeat)
        result[i, 2] /= number
        result[i, 2] *= 1000

    plt.figure(id, figsize=figure_size, dpi=80)
    plt.clf()
    plt.xticks(range(start, end + 1, step))
    plt.plot(result[:, 0, 0], np.mean(result[:, 1], axis=-1), color='b', label='Direct convolution')
    plt.fill_between(result[:, 0, 0], np.min(result[:, 1], axis=-1), np.max(result[:, 1], axis=-1), alpha=0.2,
                     edgecolor='b', facecolor='b', linewidth=1, antialiased=True)
    plt.plot(result[:, 0, 0], np.mean(result[:, 2], axis=-1), color='r', label='FFT based convolution')
    plt.fill_between(result[:, 0, 0], np.min(result[:, 2], axis=-1), np.max(result[:, 2], axis=-1), alpha=0.2,
                     edgecolor='r', facecolor='r', linewidth=1, antialiased=True)
    title = f'Computation time for a kernel size of {M}px by {N}px in function of the input signal size'
    plt.title(f"Figure {id} - {title}", fontsize=14)
    plt.ylabel('Computation time (ms)')
    plt.xlabel('Input signal size (px) square')
    plt.legend()
