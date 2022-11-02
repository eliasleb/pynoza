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
import matplotlib.pyplot as plt
import inverse_problem
import scipy
import scipy.interpolate
import itertools


def test_charge_moment_computation():
    # Moment of a dipole
    current_moment = np.zeros((3, 3, 3, 3))
    current_moment[2, 0, 0, 0] = 1
    charge_moment = inverse_problem.get_charge_moment(current_moment)
    charge_moment_analytical = np.zeros((3, 3, 3, 3))
    charge_moment_analytical[2, 0, 0, 2] = 2

    for i in {0, 1}:
        a = [0, 0, 1]
        a[i] = 1
        charge_moment_analytical[i, a[0], a[1], a[2]] = 1
    print(f"{charge_moment=}")
    print(f"{charge_moment_analytical=}")
    print(np.any(charge_moment != 0), np.any(charge_moment_analytical != 0))

    assert np.all(charge_moment == -charge_moment_analytical)

    if __name__ == "__main__":
        plt.ion()
        inverse_problem.plot_moment(current_moment)
        inverse_problem.plot_moment(charge_moment)
        inverse_problem.plot_moment(charge_moment_analytical)

        plt.pause(0.1)
        plt.show()
        input("Press Enter to continue...")


def test_inverse_problem_simple():
    """
    Test the pynoza :solution: class by solving inverse problem on a dipole source
    """
    coords = [-1., 1.]
    x1, x2, x3 = [], [], []
    for x, y, z in itertools.product(*(coords, ) * 3):
        x1.append(x), x2.append(y), x3.append(z)
    x1, x2, x3 = np.array(x1), np.array(x2), np.array(x3)

    t = np.linspace(0, 20, 100)
    wavelength = 1
    f = 1 / wavelength
    gamma = np.sqrt(12/7)/f
    t0 = 5 * gamma

    h_true = np.exp(-((t-t0)/gamma)**2) * (4 * ((t-t0)/gamma)**2 - 2)

    def get_h_num(h, t):
        h[-h.size//3:] = 0
        h[0] = 0
        return scipy.interpolate.interp1d(np.linspace(t.min(), t.max(), h.size),
                                          h,
                                          kind="cubic")(t)

    kwargs = {"tol": 1e-4,
              "n_points": 30,
              "error_tol": 5e-2,
              "coeff_derivative": 0,
              "verbose_every": 100,
              "plot": __name__ == "__main__",
              "h_num": get_h_num,
              "find_center": True,
              "scale": 1,
              "max_global_tries": 1}

    order = 2

    shape_mom = (order + 1, order + 1, order + 1, 3)
    dim_mom = 3

    def get_current_moment(moment):
        current_moment = np.zeros(shape_mom)
        current_moment[:, 0, 0, 0] = moment
        return current_moment

    e_true = direct_problem_simple(x1, x2, x3, t, h_true, order=order).swapaxes(1, 2)
    current_moment, h, center, e_opt = inverse_problem.inverse_problem(order, e_true, x1, x2, x3, t, dim_mom,
                                                                       get_current_moment, dim_mom, **kwargs)
    assert np.sum((e_true - e_opt)**2)/np.sum(e_opt**2) < 1e-2

    # if __name__ == "__main__":
    #
    #     plt.ion()
    #     print(f"{center=}")
    #
    #     inverse_problem.plot_moment(current_moment)
    #     inverse_problem.plot_moment(inverse_problem.get_charge_moment(current_moment))
    #
    #     plt.figure()
    #     h -= h[0]
    #     plt.plot(t, h/np.max(np.abs(h)))
    #     h_max = np.max(np.abs(h_true))
    #     plt.plot(t, h_true/h_max, "--")
    #     plt.xlabel("Time (relative)")
    #     plt.ylabel("Amplitude (normalized)")
    #     plt.legend(["Inverse problem", "True solution"])
    #     plt.title("Current vs time")
    #     plt.pause(0.1)
    #     plt.show()
    #     input("Press Enter to continue...")


def direct_problem_simple(x1, x2, x3, t, h, order=2):

    def current_moment(a1, a2, a3):
        if a1 == 0 and a2 == 0 and a3 == 0:
            return [0, 0, 1]
        else:
            return [0, 0, 0]

    def charge_moment(a1, a2, a3):
        a = (a1, a2, a3)
        if a == (1, 0, 1):
            return [-1, 0, 0]
        elif a == (0, 1, 1):
            return [0, -1, 0]
        elif a == (0, 0, 2):
            return [0, 0, -2]
        else:
            return [0, 0, 0]

    sol = pynoza.solution.Solution(max_order=order)
    sol.recurse()
    sol.set_moments(charge_moment=charge_moment, current_moment=current_moment)
    e = sol.compute_e_field(x1, x2, x3, t, h, None, compute_grid=False)
    return e.swapaxes(1, 2)


if __name__ == "__main__":
    test_inverse_problem_simple()
