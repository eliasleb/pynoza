import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import itertools


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


def get_charge_moment(current_moment: ndarray) -> ndarray:
    """
    Compute a charge moment that is compatible with the conservation of charge

    :param current_moment: an array with the current moments
    :return: the corresponding charge moment
    :rtype: ndarray

    The moment at index `(i, a1, a2, a3)` is the moment of the `i`-th coordinate corresponding to the multi-index
    `(a1, a2, a3)`
    """
    charge_moment = np.zeros(current_moment.shape)
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
            else:
                b[i] -= 1
                b[j] -= 1
                if a[j] >= 1 and a[i] >= 1:
                    charge_moment[i, a1, a2, a3] += a[j] * a[i] \
                        * current_moment[j, b[0], b[1], b[2]]
    return -charge_moment


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
                a[j] -= 1
                magnetic_moment[i, a1, a2, a3] += a[j] * lc * current_moment[k, a[0], a[1], a[2]]

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


def levi_civita(i: int, j: int, k: int):
    match (i, j, k):
        case (1, 2, 3) | (2, 3, 1) | (3, 1, 2):
            return 1
        case (3, 2, 1) | (1, 3, 2) | (2, 1, 3):
            return -1
        case _:
            return 0


if __name__ == "__main__":
    with PlotAndWait() as paw:
        paw.fig.add_subplot(1, 2, 1)
        plt.plot(1, 1, "x")
        paw.fig.add_subplot(1, 2, 2)
        plt.plot(1, 1, "o")
