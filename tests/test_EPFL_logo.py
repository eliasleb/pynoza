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

import numpy as np
import pandas as pd
import sympy.functions.special
import itertools
import scipy.interpolate
import pynoza
import pytest
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from pynoza import c_j


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


def legends_matrix(loc_table_x, loc_table_y, lines, line_colors, markers, marker_colors, marker_sizes, line_length=1):
    for y, line, line_color, marker, marker_color, marker_size in zip(
            loc_table_y, lines, line_colors, markers, marker_colors, marker_sizes):
        plt.plot((loc_table_x[0], loc_table_x[0] + line_length), (y, y), line, color=line_color)
        plt.scatter((loc_table_x[1],), (y,), marker=marker, color=marker_color, s=marker_size**2)


@pytest.mark.parametrize("test_case, order, method", [
    ("logo", 8, "python"),
    ("disc", 24, "python"),
    ("logo_num", 8, "python")])
def test_solution(test_case, order, method, plot=False, cname="xyz"):
    """
    Test the pynoza :solution: class by comparing
    with either COMSOL simulation (case_="logo") or
    existing literature (case_="disc")

    :param test_case: either "logo" or "disc"
    :return:
    """

    if test_case == "logo_num":
        num = True
        test_case = "logo"
    else:
        num = False

    case_ = test_case

    gamma_si = 4e-9
    gamma = 1
    c0 = 299792458. * gamma_si

    t_g = np.sqrt(7 / 12)
    wavelength = c0 * t_g

    print(f"Computing {case_=}, {order=}")
    match case_:
        case "logo":
            a = wavelength / 2
        case "disc":
            a = wavelength * 9
        case _:
            a = 1

    length = 2 * a
    length_logo = 171

    t = None
    match case_:
        case "logo":
            x1s = [0, 0, 10, 0, 0, 45, 45, 45, 45, 91, 91, 101, 91, 136, 136]
            x2s = [0, 10, 20, 30, 40, 0, 20, 30, 40, 0, 30, 20, 40, 10, 0]
            ws = [35, 10, 23, 10, 35, 10, 23, 10, 23, 10, 10, 23, 35, 10, 35]
            hs = [10, 10, 10, 10, 10, 20, 10, 10, 10, 20, 10, 10, 10, 40, 10]

            d1 = a
            d2 = a * 50 / length_logo

            w0 = 1
            h0 = 1
            x10 = np.arange(0, 171, w0)
            x20 = np.arange(0, 171, h0)
            for xi, yi in itertools.product(x10, x20):
                if 15 ** 2 >= (xi - 68) ** 2 + (yi - 35) ** 2 >= 5 ** 2 and xi >= 68:
                    x1s.append(xi - w0 / 2)
                    x2s.append(yi - h0 / 2)
                    ws.append(w0)
                    hs.append(h0)
            t = np.linspace(0, 10 * gamma, 200)

        case ("disc" | "rasc"):
            d1 = a
            d2 = a
            x1s, x2s, ws, hs = [], [], [], []
            w0 = 1
            h0 = 1
            x10 = np.arange(-2*w0, 171 + 2*w0, w0)
            x20 = np.arange(-2*h0, 171 + 2*h0, h0)
            covered_surface, n_pix = 0., 0
            for xi, yi in itertools.product(x10, x20):
                if (xi - 171 / 2) ** 2 + (yi - 171 / 2) ** 2 <= (171 / 2 + w0 / 2) ** 2 / 101.113e-2:
                    x1s.append(xi - w0 / 2)
                    x2s.append(yi - h0 / 2)
                    ws.append(w0)
                    hs.append(h0)
                    covered_surface += w0 * h0
                    n_pix += 1
            coverage = covered_surface/np.pi/(length_logo/2)**2
            print(f"Coverage: {coverage * 100:.3f} % with {n_pix=} ")

            if plot:
                plt.figure(figsize=(2.8, 2.8))
                plot_current_density(x1s, x2s, ws, hs, length_logo, length, a, a)
                plt.xlim(-1.1 * a, 1.1 * a)
                plt.ylim(-1.1 * a, 1.1 * a)
                if cname == "xyz":
                    plt.xlabel(r"$x$ (m)")
                    plt.ylabel(r"$y$ (m)")
                else:
                    plt.xlabel(r"$x_1$ (m)")
                    plt.ylabel(r"$x_2$ (m)")
                plt.tight_layout()
                plt.savefig("tests/data/disc_current_density.pdf")
            match case_:
                case "disc":
                    t = np.linspace(0, 6 * gamma, 200)
                case "rasc":
                    t = np.linspace(0, 10 * gamma, 300)
        case _:
            t = np.array([0, ])

    def current_moment(ax_, ay, az):
        moment = 0
        if az == 0:
            for x_i, y_i, wi, hi in zip(x1s, x2s, ws, hs):
                moment += c_j(ax_, ay, az, x_i / length_logo * length - d1,
                              y_i / length_logo * length - d2, wi / length_logo * length,
                              hi / length_logo * length) / gamma_si
        return [moment, 0, 0]

    x1 = np.array([0, ])
    x2 = np.array([0, ])
    match case_:
        case "logo":
            x3 = np.array((2, 3, 1000,))*a
        case ("disc" | "rasc"):
            x3 = np.array((3, 9, 18,))*a
        case _:
            x3 = np.array([0, ])

    t_sym = sympy.Symbol("t", real=True)
    h_sym = (3 * gamma_si * sympy.sqrt(np.pi / 2)) ** -.5 \
            * sympy.exp(-((t_sym-3*gamma)/gamma)**2) * (4*((t_sym-3*gamma)/gamma)**2-2)

    if num or method == "rust":
        h_sym = sympy.lambdify(t_sym, h_sym)(t)

    sol = pynoza.solution.Solution(max_order=order,
                                   wave_speed=c0)
    sol.recurse()
    current_moment_arr = np.zeros((3, order + 1, order + 1, order + 1))
    charge_moment_arr = current_moment_arr.copy()

    for ind, _ in np.ndenumerate(np.zeros((order + 1, ) * 3)):
        current_moment_arr[:, ind[0], ind[1], ind[2]] = current_moment(*ind)
    charge_moment = pynoza.helpers.get_charge_moment(current_moment_arr)
    for ind, _ in np.ndenumerate(np.zeros((order + 1, ) * 3)):
        if np.sum(ind) > order - 2:
            charge_moment_arr[:, ind[0], ind[1], ind[2]] = 0.

    sol.set_moments(current_moment=current_moment)
    e_field = None
    match case_:
        case "logo":
            match method:
                case "python":
                    e_field = sol.compute_e_field(x1, x2, x3, t, h_sym, t_sym, verbose=False)
                case "rust":
                    import speenoza
                    current_moment_array = np.zeros((3, order + 3, order + 3, order + 3))
                    for dim, a1, a2, a3 in itertools.product(range(3), *(range(order + 3), ) * 3):
                        current_moment_array[dim, a1, a2, a3] = current_moment(a1, a2, a3)[dim]
                    _x1, _x2, _x3 = [], [], []
                    for x, y, z in itertools.product(x1, x2, x3):
                        _x1.append(x)
                        _x2.append(y)
                        _x3.append(z)
                    x1, x2, x3 = np.array(_x1), np.array(_x2), np.array(_x3)
                    e_field = speenoza.multipole_e_field(x1.astype("float64").flatten(),
                                                         x2.astype("float64").flatten(),
                                                         x3.astype("float64").flatten(),
                                                         t.astype("float64").flatten(),
                                                         h_sym.astype("float64").flatten(),
                                                         current_moment_array.astype("float64")).swapaxes(1, 2)
        case "disc":
            e_field = sol.compute_e_field(x1, x2, x3, t, h_sym, t_sym, verbose=False, delayed=False)

    e_field_x = None
    match method:
        case "python":
            e_field_x = e_field[0, :, :, :, :]
        case "rust":
            e_field_x = e_field[0, :, :].reshape((1, 1, x3.size, t.size))
    if cname == "xyz":
        var_name = "z"
    else:
        var_name = "x_3"

    match case_:
        case "logo":
            filename = "applications/EPFL_logo/data/Efield_at_2a_and_3a_lambdaOver2_v3.txt"
            data_comsol = pd.read_csv(filename,
                                      skiprows=range(10),
                                      names=("t", "absE2a", "E2a", "absE3a", "E3a"),
                                      delim_whitespace=True)
            e_comsol_2a = scipy.interpolate.interp1d(data_comsol["t"], data_comsol["E2a"],
                                                     fill_value="extrapolate")(t * gamma_si)
            e_comsol_3a = scipy.interpolate.interp1d(data_comsol["t"], data_comsol["E3a"],
                                                     fill_value="extrapolate")(t * gamma_si)
            if num:
                e_field_x = pynoza.set_extremities(e_field_x, 0.1, dim=3)
                norm1 = 200
            else:
                norm1 = 160

            if plot:
                plt.subplots(figsize=(5, 3), ncols=1)
                lines = ["-", ":"]
                line_colors = ["k", "k"]
                markers = ["*", "o"]
                marker_sizes = [5, 4]
                marker_colors = ["r", "r"]

                for ind, (e_comsol, line, line_color, marker, marker_size, marker_color) in enumerate(zip(
                        [e_comsol_2a, e_comsol_3a], lines, line_colors, markers, marker_sizes, marker_colors)):
                    plt.plot(t * 4, e_comsol / 2 / 1e3, marker, markersize=marker_size,
                             color=marker_color)
                    plt.plot(t * 4, e_field_x[0, 0, ind, :] / 1e3, line, color=line_color)
                plt.legend((
                    "multipole, lambda",
                    "comsol, lambda",
                    "multipole, 1.5 lambda",
                    "comsol, 1.5 lambda"
                ))
                plt.xlim(2.5, 27.5)
                plt.ylim(-60, 100)
                plt.xlabel("Time (ns)")

                plt.savefig("tests/data/logo.pdf")
                plt.show()

            lse_2a = np.linalg.norm(e_comsol_2a / 2 - e_field_x[0, 0, 0, :], ord=2) / t.size
            lse_3a = np.linalg.norm(e_comsol_3a / 2 - e_field_x[0, 0, 1, :], ord=2) / t.size
            lse = (lse_2a + lse_3a) / 2
            print(f"{lse=}")

            assert np.linalg.norm(e_comsol_2a / 2 - e_field_x[0, 0, 0, :], ord=2) / t.size < norm1 \
                   and np.linalg.norm(e_comsol_3a / 2 - e_field_x[0, 0, 1, :], ord=2) / t.size < 170
        case "disc":
            d = pd.read_csv("applications/EPFL_logo/data/shlivinski.csv",
                            skiprows=1)
            delay = -3 * gamma_si * 1e9
            e_paper1 = scipy.interpolate.interp1d(d["X.2"] * t_g * gamma_si * 1e9 - delay, d["Y.2"],
                                                  fill_value="extrapolate")(t * gamma_si * 1e9)
            e_paper2 = scipy.interpolate.interp1d(d["X.1"] * t_g * gamma_si * 1e9 - delay, d["Y.1"],
                                                  fill_value="extrapolate")(t * gamma_si * 1e9)
            e_paper3 = scipy.interpolate.interp1d(d["X"] * t_g * gamma_si * 1e9 - delay, d["Y"],
                                                  fill_value="extrapolate")(t * gamma_si * 1e9)
            if plot:
                plt.figure(figsize=(18/2.54, 10/2.54))
                c_si = 3e8
                n_d_paper = 1
                lines2 = [(0, (1, 3)), (0, (3, 1, 1)), (0, (5, 1))]
                line_colors = ["k", "k", "k"]
                lines = [":", "--", "-"]
                marker_colors = ["r", "r", "r"]
                marker_sizes = [5, 4, 4]

                for ind, (e_paper, line, line_color, marker, marker_size, marker_color) in enumerate(zip(
                        [e_paper1, e_paper2, e_paper3], lines, line_colors, lines2, marker_sizes, marker_colors)):
                    plt.plot(t[::n_d_paper] / c_si * 1e9, e_paper[::n_d_paper] * 1e2, color=marker_color,
                             markersize=5, label="exact", linestyle=line)
                    plt.plot(t / c_si * 1e9, e_field_x[0, 0, ind, :] * x3[ind] / 1e6, linestyle=marker,
                             label="approx.", color=line_color,)

                plt.text(1, 0, f"{order=}")

                plt.legend()
                plt.xlim(0, 20)
                plt.xlabel("Time (ns)")
                plt.ylabel("Voltage (MV)")
                plt.tight_layout()
                plt.savefig("tests/data/test_approximate_vs_exact.pdf")
                plt.show()

            assert np.linalg.norm(e_paper1 * 1e2 - e_field_x[0, 0, 0, :] * x3[0] / 1e6, ord=2) / t.size < 0.4 \
                   and np.linalg.norm(e_paper2 * 1e2 - e_field_x[0, 0, 1, :] * x3[1] / 1e6, ord=2) / t.size < 0.4 \
                   and np.linalg.norm(e_paper3 * 1e2 - e_field_x[0, 0, 2, :] * x3[2] / 1e6, ord=2) / t.size < 0.4

    with open(f"tests/data/field-{case_}.txt", "w+") as fd:
        fd.write(sol.get_e_field_text())


def plot_lse_logo():

    orders = range(11)
    lses = [
        1663.5875518751154,
        1122.4116072604822,
        370.113466300506,
        370.113466300506,
        187.79797208479295,
        187.79797208479295,
        164.90432083428794,
        164.90432083428794,
        162.29050193127722,
        162.29050193127722,
        162.23040795771308
    ]
    plt.figure(figsize=(5, 3))
    plt.plot(np.array(orders) + 2, 20 * np.log10(np.array(lses)/1e3), "ko-")

    plt.ylabel(r"Least-square error (dBkV)")
    plt.xlabel("Truncation order $n$")

    plt.tight_layout()
    plt.savefig("tests/data/lse_logo.pdf")


if __name__ == "__main__":
    import argparse

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times"
    })

    plot_lse_logo()

    parser = argparse.ArgumentParser()
    parser.add_argument("--order", metavar="order", type=int, required=True)
    parser.add_argument("--case", metavar="case", type=str, choices=["logo", "logo_num", "disc"], required=True)
    parser.add_argument("--method", metavar="method", type=str, choices=["python", "rust"], required=True)
    parser.add_argument("--cname", metavar="cname", type=str, choices=["xyz", "x1x2x3"], required=False,
                        default="xyz")

    args = parser.parse_args()
    test_solution(args.case, args.order, args.method, plot=True, cname=args.cname)
