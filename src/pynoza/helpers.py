import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray


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

    The moment at index `(i, a1, a2, a3)` is the moment of the `i`th coordinate corresponding to the multi-index
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


if __name__ == "__main__":
    with PlotAndWait() as paw:
        paw.fig.add_subplot(1, 2, 1)
        plt.plot(1, 1, "x")
        paw.fig.add_subplot(1, 2, 2)
        plt.plot(1, 1, "o")
