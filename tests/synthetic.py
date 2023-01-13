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


import numpy as np
import pynoza
import matplotlib.pyplot as plt
import itertools
import inverse_problem


def build_current_moment(order):
    current_moment = np.zeros((3, order + 3, order + 3, order + 3))
    for i in range(order + 1):
        current_moment[2, 0, 0, i] = (-1) ** i / (i + 1)
    return current_moment


def plot_directivity(solution, r, h, t, phi_min=0, phi_max=2*np.pi):
    theta = np.linspace(0, np.pi, 30)
    phi = np.linspace(phi_min, phi_max, 60)
    coords_directivity = [[], [], []]
    for theta_i, phi_i in itertools.product(theta, phi):
        coords_directivity[0].append(r * np.sin(theta_i) * np.cos(phi_i))
        coords_directivity[1].append(r * np.sin(theta_i) * np.sin(phi_i))
        coords_directivity[2].append(r * np.cos(theta_i))

    coords_directivity = np.array(coords_directivity)
    e_pred = solution.compute_e_field(coords_directivity[0], coords_directivity[1], coords_directivity[2],
                                      t, h, None, compute_grid=False)

    with pynoza.PlotAndWait(new_figure=True):
        plt.plot(t, e_pred[2, :, :].T)

    energy = np.sum(e_pred**2, axis=(0, 2))
    energy = energy / np.max(energy)

    with pynoza.PlotAndWait(new_figure=False):
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(coords_directivity[0] / r * energy,
                   coords_directivity[1] / r * energy,
                   coords_directivity[2] / r * energy, color="b")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")


def synthetic_antenna():
    order = 11
    current_moment = build_current_moment(order)
    current_moment_callable = lambda a1, a2, a3: list(current_moment[:, a1, a2, a3])

    sol = pynoza.Solution(max_order=order + 2)
    sol.recurse()
    sol.set_moments(current_moment=current_moment_callable)
    t = np.linspace(0, 5, 100)
    f = 4
    gamma = np.sqrt(12 / 7) / f
    t0 = 3 * gamma
    h = np.exp(-((t - t0) / gamma)**2) * (4 * ((t - t0) / gamma)**2 - 2)
    r = 2.5
    plot_directivity(sol, r, h, t)


if __name__ == "__main__":
    synthetic_antenna()
