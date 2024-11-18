import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import hashlib
import re
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from numpy.linalg import lstsq
from scipy.optimize import minimize


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


def cache_function_call(func, *args, cache_dir="function_cache", raise_error_if_not_cached=False, **kwargs):
    """
    Caches the result of a function call and retrieves it from the cache if the same call is made again.

    Args:
    :param func: The function to call.
    :param args: Positional arguments of the function.
    :param cache_dir: Directory used to cache function calls
    :param raise_error_if_not_cached: If true, raises FileNotFoundError if the call is not cached instead of calling
        :func:
    :param kwargs: Keyword arguments of the function.

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
    elif raise_error_if_not_cached:
        raise FileNotFoundError(f"Hash {hash_key} not found. Set :raise_error_if_not_cached: to False to disable")

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

    rectangles = [Rectangle((_x / length_logo * length - d1, y / length_logo * length - d2),
                            w / length_logo * length, h / length_logo * length) for _x, y, w, h in zip(xs, ys, ws, hs)]
    pc = PatchCollection(rectangles, facecolor="r", **kwargs)
    if ax is None:
        ax = plt.gca()
    ax.add_collection(pc)


def get_poynting(e, h):
    """Returns the Poynting vector E x H
    #TODO the usual checks
    """
    p = np.zeros(e.shape)
    p[0, ...] = e[1, ...] * h[2, ...] - e[2, ...] * h[1, ...]
    p[1, ...] = e[2, ...] * h[0, ...] - e[0, ...] * h[2, ...]
    p[2, ...] = e[0, ...] * h[1, ...] - e[1, ...] * h[0, ...]
    return p


def lstsq_moment_problem_solution(z, moments, n_steps=10, rcond=-1):
    if not isinstance(moments, np.ndarray) or len(moments.shape) != 1:
        raise ValueError(":moments: must be a 1d numpy array")
    if not isinstance(z, np.ndarray) or len(z.shape) != 1:
        raise ValueError(":z: must be a 1d numpy array")

    max_order = moments.size - 1

    matrix = np.zeros((max_order + 1, n_steps))
    for ind_l, l in enumerate(range(0, max_order + 1)):
        for ind_fun, fun in enumerate(range(1, n_steps + 1)):
            matrix[ind_l, ind_fun] = (10 ** (-1 - l) * (1 + (-1) ** l) * (11 - fun) ** (1 + l)) / (1 + l)

    sol, residual, _, _ = lstsq(matrix, moments, rcond=rcond)
    fit = 0.
    heaviside_pi = lambda y: np.abs(y) < .5
    for fun_number, sol_i in zip(range(1, n_steps + 1), reversed(sol)):
        fit = fit + heaviside_pi(z / 2 * n_steps / fun_number) * sol_i
    return fit, residual


def optimization_moment_problem_solution(z, moments, poly_order=2, **opt_kwargs):
    if not isinstance(moments, np.ndarray) or len(moments.shape) != 1:
        raise ValueError(":moments: must be a 1d numpy array")
    if not isinstance(z, np.ndarray) or len(z.shape) != 1:
        raise ValueError(":z: must be a 1d numpy array")

    should_be_negative = opt_kwargs.pop("should_be_negative", False)
    should_be_positive = opt_kwargs.pop("should_be_positive", False)
    weight_l2 = opt_kwargs.pop("weight_l2", 0.)

    dz = z[1] - z[0]

    def get_poly(param):
        poly = 0.

        if poly_order == 2:
            # params are samples at 0, 1/2, 1
            param = [
                param[0],
                -3 * param[0] + 4 * param[1] - param[2],
                2 * (param[0] - 2 * param[1] + param[2])
            ]

        for p_order, coeff in zip(range(0, poly_order + 1), param):
            poly = poly + coeff * z ** p_order
        poly[z < 0] = poly[z > 0][:np.sum(z < 0)][::-1]
        return poly

    def get_error(param):
        result = 0.
        poly = get_poly(param)
        for ind, moment_i in enumerate(moments):
            opt_moment_i = dz * np.sum(z**ind * poly)
            result += (opt_moment_i - moment_i) ** 2
        result += weight_l2 * dz * np.sum(poly**2)
        if should_be_negative:
            return result + np.sum(poly[poly > 0])
        if should_be_positive:
            return result - np.sum(poly[poly < 0])
        return result

    sol = minimize(get_error, np.zeros((poly_order + 1, )), **opt_kwargs)
    y = get_poly(sol.x)
    return y, sol.fun


if __name__ == "__main__":

    def small_test():
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

    small_test()
