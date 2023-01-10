#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022  Elias Le Boudec
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.


import pynoza
import numpy as np


def toy_current_density(_a1: int, a2: int, a3: int) -> list[float, float, float]:
    """
    Toy example current density moment computation.

    :param _a1: multi-index, first component
    :param a2: multi-index, second component
    :param a3: multi-index, third component
    :return: the current moments for every component
    """
    return [10**-(a2 + 5), 0, (-1)**a2 * 10**-a3]


def test_simple_example():
    max_order = 10

    current_moment_array = np.zeros((3, max_order + 2, max_order + 2, max_order + 2))
    for ind, _ in np.ndenumerate(np.zeros(current_moment_array.shape[1:])):
        current_moment_array[:, ind[0], ind[1], ind[2]] = toy_current_density(ind[0], ind[1], ind[2])
    charge_moment_array = pynoza.get_charge_moment(current_moment_array)

    current_moment = lambda a1, a2, a3: list(current_moment_array[:, a1, a2, a3])
    charge_moment = lambda a1, a2, a3: list(charge_moment_array[:, a1, a2, a3])

    sol = pynoza.Solution(max_order)
    sol.recurse()
    sol.set_moments(current_moment, charge_moment)

    wavelength = 1
    x1 = np.array((wavelength, 2 * wavelength, ))
    x2 = np.zeros((1, ))
    x3 = np.array((-wavelength, wavelength, ))

    t = np.linspace(0, 4, 1000)
    h = np.exp(-(t - 1)**2 * 100)

    e_field = sol.compute_e_field(x1, x2, x3, t, h, None)

    return t, h, e_field


def show_results():
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("TkAgg")

    t, h, e_field = test_simple_example()

    plt.plot(t, h)
    plt.xlabel("Time")
    plt.ylabel("Current shape $h$")
    plt.tight_layout()
    plt.xlim(np.min(t), np.max(t))

    plt.figure(figsize=(9, 6))

    overall_max = np.max(np.abs(e_field))

    for component in range(3):
        plt.subplot(3, 1, component + 1)
        plt.plot(t, e_field[component, :, 0, :, :].reshape(4, t.size).T)
        plt.ylim(-overall_max, overall_max)
        plt.xlim(np.min(t), np.max(t))
        plt.xlabel("Time")
        plt.ylabel("E-field")
        plt.title(f"E-field, component #{component + 1} at each point")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_results()
