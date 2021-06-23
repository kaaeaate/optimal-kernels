import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from pyefd import elliptic_fourier_descriptors

def plot_efd(coeffs, order, locus=(0.0, 0.0), image=None, contour=None):
    """Plot a ``[2 x (N / 2)]`` grid of successive truncations of the series.

    .. note::

        Requires `matplotlib <http://matplotlib.org/>`_!

    :param numpy.ndarray coeffs: ``[N x 4]`` Fourier coefficient array.
    :param list, tuple or numpy.ndarray locus:
        The :math:`A_0` and :math:`C_0` elliptic locus in [#a]_ and [#b]_.
    :param int n: Number of points to use for plotting of Fourier series.

    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Cannot plot: matplotlib was not installed.")
        return

    N = coeffs.shape[0]
    N_half = int(np.ceil(N / 2))
    n_rows = 2
    n_cols = order//n_rows

    t = np.linspace(0, 1.0, order)
    xt = np.ones((order,)) * locus[0]
    yt = np.ones((order,)) * locus[1]
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(15,8))
    for n in range(coeffs.shape[0]):
        yt += (coeffs[n, 0] * np.cos(2 * (n + 1) * np.pi * t)) + (
            coeffs[n, 1] * np.sin(2 * (n + 1) * np.pi * t)
        )
        xt += (coeffs[n, 2] * np.cos(2 * (n + 1) * np.pi * t)) + (
            coeffs[n, 3] * np.sin(2 * (n + 1) * np.pi * t)
        )
        if n < n_cols:
            line = 0
            k = n
        else:
            line = 1
            k = n-n_cols
        ax[line, k].plot(yt, xt, 'r', linewidth=2)
        if image is not None:
            ax[line, k].imshow(image)

    plt.show()