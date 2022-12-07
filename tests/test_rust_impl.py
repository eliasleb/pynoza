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

import pytest
import speenoza
import numpy as np
import pynoza
import matplotlib.pyplot as plt
import inverse_problem
import itertools


def test_syntax():
    max_order = 2
    x1 = np.array([1, 0, 0, ], dtype=float)
    x2 = np.array([0, 1, 0, ], dtype=float)
    x3 = np.array([0, 0, 1, ], dtype=float)
    t = np.linspace(0, 10, 100, dtype=float)
    h = np.sin(2 * np.pi * t)
    moment = np.zeros((3, max_order + 1, max_order + 1, max_order + 1, ))
    moment[2, 0, 0, 0] = 1.
    sol = speenoza.multipole_e_field(x1, x2, x3, t, h, moment)


def test_inputs():
    x1 = np.array([1, 0, ], dtype=float)
    x2 = np.array([0, 0, 0, ], dtype=float)
    x3 = np.array([0, 0, 0, ], dtype=float)
    t = np.linspace(0, 10, 10, dtype=float)
    h = np.sin(2 * np.pi * t)
    moment = np.zeros((3, 1, 1, 1, ))
    moment[2, 0, 0, 0] = 1.

    with pytest.raises(ValueError):
        speenoza.multipole_e_field(x1, x2, x3, t, h, moment)

    x1 = np.array([1, 0, 0, ], dtype=float)
    h = h[1:]
    with pytest.raises(ValueError):
        speenoza.multipole_e_field(x1, x2, x3, t, h, moment)


def test_rust_vs_python_simple_case():
    x1 = np.linspace(.5, 10, 6)
    x2 = np.zeros((x1.size, ), dtype=float)
    x3 = np.zeros((x1.size, ), dtype=float)
    t = np.linspace(0, 20, 2_000)
    t0 = 3
    gamma = 1
    h = np.exp(-((t - t0)/gamma)**2) * (4 * ((t - t0)/gamma)**2 - 2)
    dim = 3
    moment = np.zeros((3, dim, dim, dim, ))
    moment[2, 0, 0, 0] = 1.
    field1 = speenoza.multipole_e_field(x1, x2, x3, t, h, moment)
    sol = pynoza.Solution(max_order=dim - 1, wave_speed=1)
    sol.recurse()
    charge_moment = pynoza.get_charge_moment(moment)
    sol.set_moments(charge_moment=lambda a1, a2, a3: list(charge_moment[:, a1, a2, a3]),
                    current_moment=lambda a1, a2, a3: list(moment[:, a1, a2, a3]))
    field2 = sol.compute_e_field(x1, x2, x3, t, h, None, delayed=True, compute_grid=False)
    error = np.sum((field1.swapaxes(1, 2) - field2)**2) / np.sum(field2**2)
    assert error < 0.0033


def plot_simple_field():
    x1 = np.linspace(.5, 10, 6)
    x2 = np.zeros((x1.size, ), dtype=float)
    x3 = np.zeros((x1.size, ), dtype=float)
    t = np.linspace(0, 20, 1000)
    t0 = 3
    gamma = 1
    h = np.exp(-((t - t0)/gamma)**2) * (4 * ((t - t0)/gamma)**2 - 2)
    with pynoza.PlotAndWait():
        plt.plot(t, h)
    plt.clf()
    dim = 3
    moment = np.zeros((3, dim, dim, dim, ))
    for i in range(dim - 2):
        moment[2, 0, i, 0] = (-1.)**i / (i + 1)**1
    print("rust sol...")
    field1 = speenoza.multipole_e_field(x1, x2, x3, t, h, moment)
    print("done")
    sol = pynoza.Solution(max_order=dim - 1, wave_speed=1)
    sol.recurse()
    charge_moment = pynoza.get_charge_moment(moment)
    sol.set_moments(charge_moment=lambda a1, a2, a3: list(charge_moment[:, a1, a2, a3]),
                    current_moment=lambda a1, a2, a3: list(moment[:, a1, a2, a3]))
    print("Python sol...")
    field2 = sol.compute_e_field(x1, x2, x3, t, h, None, delayed=True, compute_grid=False)
    print("done")
    dt = t[1] - t[0]
    h = h.reshape((h.size, 1))
    p = np.cumsum(h, axis=0) * dt
    dp = np.concatenate((np.diff(p, axis=0) / dt, np.zeros((1, 1))))
    ddp = np.concatenate((np.diff(dp, axis=0) / dt, np.zeros((1, 1))))
    r = np.sqrt(x1**2 + x2**2 + x3**2).reshape((1, x1.size))
    field3 = -1 / 4 / np.pi * (p / r**2 + dp / r + ddp) * 1.4e-6 * 7.17 / 8.02
    # with pynoza.PlotAndWait():
    #     plt.plot(p)
    #     plt.plot(dp)
    #     plt.plot(ddp)
    # plt.clf()
    plt_lim = np.max(np.abs(field2))
    with pynoza.PlotAndWait():
        plt.plot(t, field1[2, :, :] * r.reshape((1, x1.size)))
        plt.plot(t, field2[2, :, :].T * r.reshape((1, x1.size)), '--')
        #plt.ylim(-plt_lim, plt_lim)
        #plt.plot(t, field3, '-.')


if __name__ == "__main__":
    plot_simple_field()
