import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import itertools
import os
import pickle
import hashlib
import re
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


class PlotAndWait:
    def __init__(self, *args, **kwargs):
        self.wait_for_enter_keypress = kwargs.pop("wait_for_enter_keypress", True)
        self.new_figure = kwargs.pop("new_figure", False)
        plt.ion()
        if self.new_figure:
            self.fig = plt.figure(*args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.pause(0.0001)
        plt.show()
        if self.wait_for_enter_keypress:
            input("[Enter] to continue...")


def get_charge_moment(current_moment: ndarray, return_mapping=False) -> ndarray:
    """
    Compute a charge moment that is compatible with the conservation of charge

    :param current_moment: an array with the current moments
    :param return_mapping: return a dict to keep a track of which charge moments correspond to which current moments
    :return: the corresponding charge moment (+ eventually the charge-current mapping)

    The moment at index `(i, a1, a2, a3)` is the moment of the `i`-th coordinate corresponding to the multi-index
    `(a1, a2, a3)`
    """
    charge_moment = np.zeros(current_moment.shape)
    mapping = {}
    for ind, _ in np.ndenumerate(charge_moment):
        i, a1, a2, a3 = ind
        a = (a1, a2, a3)
        for j in range(3):
            b = list(a)
            if i == j:
                if a[j] >= 2:
                    b[j] = a[j] - 2
                    charge_moment[i, a1, a2, a3] += a[j] * (a[j] - 1) \
                        * current_moment[j, b[0], b[1], b[2]]
                    if a not in mapping:
                        mapping[tuple(ind)] = set()
                    mapping[tuple(ind)].add((j, ) + tuple(b))
            else:
                b[i] -= 1
                b[j] -= 1
                if a[j] >= 1 and a[i] >= 1:
                    charge_moment[i, a1, a2, a3] += a[j] * a[i] \
                        * current_moment[j, b[0], b[1], b[2]]
                    if a not in mapping:
                        mapping[tuple(ind)] = set()
                    mapping[tuple(ind)].add((j, ) + tuple(b))
    charge_moment = -charge_moment
    if return_mapping:
        return charge_moment, mapping
    return charge_moment


def get_magnetic_moment(current_moment: ndarray) -> ndarray:
    """
    Compute the magnetic moment corresponding to the curl of the current density from the current moments.

    :param current_moment: an array with the current moments
    :return: the corresponding magnetic moment
    :rtype: ndarray

    The moment at index `(i, a1, a2, a3)` is the moment of the `i`-th coordinate corresponding to the multi-index
    `(a1, a2, a3)`
    """
    magnetic_moment = np.zeros(current_moment.shape)
    for ind, _ in np.ndenumerate(magnetic_moment):
        i, a1, a2, a3 = ind
        a = [a1, a2, a3]
        for j, k in itertools.product(range(3), repeat=2):
            lc = levi_civita(i, j, k)
            if a[j] > 0 and lc != 0:
                a_copy = [a1, a2, a3]
                a_copy[j] -= 1
                magnetic_moment[i, a1, a2, a3] += a[j] * lc * current_moment[k, a_copy[0], a_copy[1], a_copy[2]]

    return -magnetic_moment


def int_j(x0, sig, m):
    """
    Computes a current space-pulse moment.

    :param x0: center of the pulse
    :param sig: width of the pulse
    :param m: order of the moment
    :return: current moment
    """
    return 1 / (m + 1) * ((x0 + sig / 2) ** (m + 1) - (x0 - sig / 2) ** (m + 1))


def int_r(x0, sig, m):
    """
    Computes a charge space-pulse moment

    :param x0: center of the pulse
    :param sig: width of the pulse
    :param m: order of the moment
    :return: charge moment
    """
    if m < 2:
        return 0
    else:
        return -m * ((sig / 2 + x0) ** (m - 1) - (-sig / 2 + x0) ** (m - 1))


def c_j(a1, a2, _a3, x1, x2, w, h):
    """
    Compute a current moment

    :param a1: multi-index, first dimension
    :param a2: multi-index, second dimension
    :param _a3: multi-index, third dimension
    :param x1: first coordinate of the rectangle center
    :param x2: second coordinate of the rectangle center
    :param w: width of the rectangle (first coordinate)
    :param h: height of the rectangle (second coordinate)
    :return: the current moment
    """
    x1 += w / 2
    x2 += h / 2
    return int_j(x1, w, a1) * int_j(x2, h, a2)


def c_r(a1, a2, _a3, x1, x2, w, h):
    """
    Compute a current moment

    :param a1: multi-index, first dimension
    :param a2: multi-index, second dimension
    :param _a3: multi-index, third dimension
    :param x1: first coordinate of the rectangle center
    :param x2: second coordinate of the rectangle center
    :param w: width of the rectangle (first coordinate)
    :param h: height of the rectangle (second coordinate)
    :return: the charge moment
    """
    x1 += w / 2
    x2 += h / 2
    return int_r(x1, w, a1) * int_j(x2, h, a2)


def levi_civita(i: int, j: int, k: int, start_at_0=True):
    if start_at_0:
        i, j, k = i + 1, j + 1, k + 1
    match (i, j, k):
        case (1, 2, 3) | (2, 3, 1) | (3, 1, 2):
            return 1
        case (3, 2, 1) | (1, 3, 2) | (2, 1, 3):
            return -1
        case _:
            return 0


def cache_function_call(func, *args, cache_dir="function_cache", **kwargs):
    """
    Caches the result of a function call and retrieves it from the cache if the same call is made again.

    Args:
    :func (function): The function to call.
    :*args: Positional arguments of the function.
    :cache_dir: Directory used to cache function calls
    :**kwargs: Keyword arguments of the function.

    Returns:
    The result of the function call.
    """

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    args_txt = str(args)
    kwargs_txt = str(kwargs)
    pattern = r' at 0x[0-9a-fA-F]+>'
    replacement = ' at 0x***>'
    args_txt = re.sub(pattern, replacement, args_txt)
    kwargs_txt = re.sub(pattern, replacement, kwargs_txt)

    args_hash = pickle.dumps((func.__name__, args_txt, kwargs_txt))
    hash_key = hashlib.sha256(args_hash).hexdigest()
    print(f"Function call with hash {hash_key}")
    cache_path = os.path.join(cache_dir, hash_key + '.pkl')

    # Check if the result is cached
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            result = pickle.load(f)
        print("Retrieved from cache.")
        return result

    # Compute the result and cache it
    result = func(*args, **kwargs)
    with open(cache_path, 'wb') as f:
        pickle.dump(result, f)

    return result


def plot_current_density(xs: list[float, ...], ys: list[float, ...], ws: list[float, ...], hs: list[float, ...],
                         length_logo, length, d1, d2, ax=None, **kwargs):
    """
    Plot a given current density

    :param xs: x-coordinates of all current rectangle lower right corner
    :param ys: y-coordinates of all current rectangle lower right corner
    :param ws: widths of all current rectangle
    :param hs: heights of all current rectangle
    :param length_logo: width of the logo in xs/ys units
    :param length: width of the logo in true units
    :param d1: physical width of logo
    :param d2: physical height of logo
    :param ax: optional axis to use
    """

    rectangles = [Rectangle((x / length_logo * length - d1, y / length_logo * length - d2),
                            w / length_logo * length, h / length_logo * length) for x, y, w, h in zip(xs, ys, ws, hs)]
    pc = PatchCollection(rectangles, facecolor="r", **kwargs)
    if ax is None:
        ax = plt.gca()
    ax.add_collection(pc)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("TkAgg")
    with PlotAndWait(new_figure=True) as paw:
        paw.fig.add_subplot(1, 2, 1)
        plt.plot(1, 1, "x")
        paw.fig.add_subplot(1, 2, 2)
        plt.plot(1, 1, "o")

    import time

    def f(_x):
        time.sleep(2)
        return _x**2

    x = 2

    def call():
        return cache_function_call(f, x)

    print(call())
    print(call())
