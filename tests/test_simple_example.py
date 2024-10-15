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
import pynoza.helpers
import numpy as np


def toy_current_density(_a1: int, a2: int, a3: int) -> list[float, float, float]:
    """
    Toy example current density moment computation.

    :param _a1: multi-index, first component
    :param a2: multi-index, second component
    :param a3: multi-index, third component
    :return: the current moments for every component
    """
    if _a1 == 0 and a2 == 0 and a3 == 0:
        return [0, 0, 1]
    return [0, 0, 0]


def test_simple_example(return_result=False, do_assert=True):
    max_order = 2

    current_moment_array = np.zeros((3, max_order + 2, max_order + 2, max_order + 2))
    for ind, _ in np.ndenumerate(np.zeros(current_moment_array.shape[1:])):
        current_moment_array[:, ind[0], ind[1], ind[2]] = toy_current_density(ind[0], ind[1], ind[2])

    current_moment = lambda a1, a2, a3: list(current_moment_array[:, a1, a2, a3])

    sol = pynoza.Solution(max_order, wave_speed=3e8)
    sol.recurse()
    sol.set_moments(current_moment)

    distance = 3
    x1 = np.array((distance, 0, 0))
    x2 = np.array((0, distance, 0))
    x3 = np.array((0, 0, distance))

    t = np.linspace(0/3e8, 7/3e8, 1000)
    dt = t[1] - t[0]
    t0, sigma = 1 / 3e8, 1 / (10*3e8)
    h = np.exp(-(t - t0)**2 / sigma**2)
    r = np.sqrt(x1**2 + x2**2 + x3**2)
    rho = np.sqrt(x1**2 + x2**2)
    theta = np.arctan2(rho, x3)
    h2 = np.exp(-(t[None, :] - t0 - r[:, None]/3e8)**2 / sigma**2)
    dh_dt = np.diff(h2, axis=1) / dt
    dh_dt = np.concatenate([np.zeros((3, 1)), dh_dt], axis=1)
    e_theta = -1 / 4 / np.pi / 8.854e-12 * np.sin(theta[:, None]) / r[:, None] / 3e8**2 * dh_dt
    b_phi = -1 / 4 / np.pi * np.sin(theta[:, None]) / r[:, None] / 3e8 * dh_dt * sol.get_mu()

    e_field = sol.compute_e_field(x1, x2, x3, t, h, None, compute_grid=False)
    b_field = sol.compute_b_field(x1, x2, x3, t, h, None, compute_grid=False)
    poynting = 1 / sol.get_mu() * np.cross(e_field, b_field, axis=0)
    # rtol = 1
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use("TkAgg")
    # plt.plot(e_field[2, 0, :])
    # plt.plot(e_theta[0, :])
    # plt.show()
    if do_assert:
        assert np.allclose(e_field[2, 0, :], e_theta[0, :], atol=5)
        assert np.allclose(e_field[2, 1, :], e_theta[1, :], atol=5)
        assert np.allclose(e_field[2, 2, :], e_theta[2, :], atol=10)
        assert np.allclose(b_field[1, 0, :], -b_phi[0, :], atol=2e-8)
        assert np.allclose(b_field[0, 1, :], b_phi[1, :], atol=2e-8)
        assert np.allclose(b_field[2, 2, :], b_phi[2, :], atol=1e-8)

    if return_result:
        return t, h, e_field, b_field, poynting, e_theta, b_phi


def show_results():
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("TkAgg")
    t, h, e_field, b_field, _poynting, e_theta, b_phi = test_simple_example(return_result=True, do_assert=False)

    plt.plot(t, h)
    plt.xlabel("Time")
    plt.ylabel("Current shape $h$")
    plt.tight_layout()
    plt.xlim(np.min(t), np.max(t))

    def plot(_t, field, comp):
        plt.figure(figsize=(10, 10))

        overall_max = np.max(np.abs(field))
        colors = ("r", "g", "b")

        for component in range(3):
            for ind, c in enumerate(colors):
                plt.subplot(3, 3, 3*component + 1 + ind)
                plt.plot(_t, field[component, ind, :], color="k", label=f"Mult. P{ind + 1}")
                plt.plot(_t, comp[ind, :].T, ":", color="r", label=f"FF P{ind + 1}")
                plt.legend(loc="upper right")
                plt.ylim(-1.1 * overall_max, 1.1 * overall_max)
                plt.xlim(np.min(t), np.max(t))
                plt.xlabel("Time")
                plt.title(f"Field component #{component + 1} at P{ind+1}")

        plt.tight_layout()

    plot(t, e_field, e_theta)
    plot(t, b_field, b_phi)
    plot(t, _poynting, e_theta)

    plt.show()


def test_charge_inversion():
    max_order = 2

    current_moment_array = np.zeros((3, max_order + 2, max_order + 2, max_order + 2))
    for ind, _ in np.ndenumerate(np.zeros(current_moment_array.shape[1:])):
        if np.sum(ind) <= max_order:
            current_moment_array[:, ind[0], ind[1], ind[2]] = toy_current_density(ind[0], ind[1], ind[2])

    _, mapping = pynoza.helpers.get_charge_moment(current_moment_array, return_mapping=True)


if __name__ == "__main__":
    show_results()
    # test_charge_inversion()
