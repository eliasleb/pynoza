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

import inverse_problem
import pynoza
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import itertools


def plot_directivity(alpha, fig, subplot, axis):
    theta = np.linspace(0, np.pi, 30)
    phi = np.linspace(0, 2*np.pi, 60)

    c0 = 3e8
    f = 500e6 / c0
    gamma = np.sqrt(12 / 7) / f
    t0 = 3 * gamma
    wavelength = 1 / f
    r = 10 * wavelength
    t_max = 6 * gamma + r
    t = np.linspace(0, t_max, 150)
    t0 = 3 * gamma
    h = np.exp(-((t - t0) / gamma)**2) * (4 * ((t - t0)/gamma)**2 - 2)

    x1 = np.zeros((theta.size * phi.size, ))
    x2, x3 = x1.copy(), x1.copy()
    for i, (ti, pi) in enumerate(itertools.product(theta, phi)):
        x1[i] = r * np.sin(ti) * np.cos(pi)
        x2[i] = r * np.sin(ti) * np.sin(pi)
        x3[i] = r * np.cos(ti)

    order = int(np.sum(alpha))

    sol = pynoza.Solution(max_order=order + 2)
    sol.recurse()

    current_moment_array = np.zeros((3, order + 3, order + 3, order + 3))
    current_moment_array[axis, alpha[0], alpha[1], alpha[2]] = 1
    charge_moment_array = pynoza.get_charge_moment(current_moment_array)

    current_moment = lambda a1, a2, a3: list(current_moment_array[:, a1, a2, a3])
    charge_moment = lambda a1, a2, a3: list(charge_moment_array[:, a1, a2, a3])

    sol.set_moments(current_moment=current_moment, charge_moment=charge_moment)
    field = sol.compute_e_field(x1, x2, x3, t, h, None, compute_grid=False)

    energy = np.sum(field**2, axis=(0, 2))

    ax = fig.add_subplot(*subplot, projection='3d')

    energy_max = energy.max()
    energy = energy / energy_max
    field_max = field.max(axis=(0, 2))
    field_min = field.min(axis=(0, 2))

    plt_x, plt_y, plt_z, colors = [], [], [], []

    for xi, yi, zi, ei, fmax, fmin in zip(x1, x2, x3, energy, field_max, field_min):
        ti = np.arctan2(np.sqrt(xi**2 + yi**2), zi)
        pi = np.arctan2(yi, xi)
        plt_x.append(ei * np.sin(ti) * np.cos(pi))
        plt_y.append(ei * np.sin(ti) * np.sin(pi))
        plt_z.append(ei * np.cos(ti))
        if abs(fmax) > abs(fmin):
            colors.append([0, 0, ei])
        else:
            colors.append([ei, 0, 0])

    ax.scatter(plt_x, plt_y, plt_z, c=colors)
   # ax.set_xlabel("x")
   # ax.set_ylabel("y")
   # ax.set_zlabel("z")
   # ax.xaxis.set_ticklabels([])
   # ax.yaxis.set_ticklabels([])
   # ax.zaxis.set_ticklabels([])
    plt.axis("off")
    plt.title(f"{alpha}")


if __name__ == "__main__":
    max_order = 4
    n_cols = (max_order + 1) * (max_order + 2) // 2

    def plot_comp(axis):
        fig = plt.figure(figsize=(29.7 / 2.54, 21 / 2.54))
        index = 0
        indices_subplot = [itertools.count(i) for i in range(1, n_cols * (max_order + 1) + 1, n_cols)]
        for i, j, k in itertools.product(range(max_order + 1), range(max_order + 1), range(max_order + 1)):
            if i + j + k <= max_order:
                plot_directivity([i, j, k], fig, (max_order + 1, n_cols, indices_subplot[i + j + k].__next__()), axis)
        plt.tight_layout()

    for i in range(3):
        plot_comp(i)
        plt.savefig(f"../../../logs/data/2022_06/directivity_comp_{i}.pdf")
    plt.show()
