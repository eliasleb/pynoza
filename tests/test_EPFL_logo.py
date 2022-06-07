#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sympy.functions.special
import itertools
import scipy.interpolate
import pynoza
import pytest


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


def c_j(a1, a2, a3, x1, x2, w, h):
    """
    Compute a current moment

    :param a1: multi-index, first dimension
    :param a2: multi-index, second dimension
    :param a3: multi-index, third dimension
    :param x1: first coordinate of the rectangle center
    :param x2: second coordinate of the rectangle center
    :param w: width of the rectangle (first coordinate)
    :param h: height of the rectangle (second coordinate)
    :return: the current moment
    """
    x1 += w / 2
    x2 += h / 2
    return int_j(x1, w, a1) * int_j(x2, h, a2)


def c_r(a1, a2, a3, x1, x2, w, h):
    """
    Compute a current moment

    :param a1: multi-index, first dimension
    :param a2: multi-index, second dimension
    :param a3: multi-index, third dimension
    :param x1: first coordinate of the rectangle center
    :param x2: second coordinate of the rectangle center
    :param w: width of the rectangle (first coordinate)
    :param h: height of the rectangle (second coordinate)
    :return: the charge moment
    """
    x1 += w / 2
    x2 += h / 2
    return int_r(x1, w, a1) * int_j(x2, h, a2)


@pytest.mark.parametrize("test_case", ["logo", "disc"])
def test_solution(test_case):
    """
    Test the pynoza :solution: class by comparing
    with either COMSOL simulation (case_="logo") or
    existing literature (case_="disc")

    :param test_case: either "logo" or "disc"
    :return:
    """

    case_ = test_case

    mu = 4 * np.pi * 1e-7
    gamma_SI = 4e-9
    gamma = 1
    c0 = 299792458 * gamma_SI

    Tg = np.sqrt(7 / 12)
    wavelength = c0 * Tg

    print(f"Computing {case_=}")
    match case_:
        case "logo":
            a = wavelength / 2
        case "disc":
            a = wavelength * 9
        case _:
            a = 1

    L = 2 * a
    Llogo = 171

    match case_:
        case "logo":
            margin = 0.1
            #      1.  2.  3.  4.  5.  6.  7.  8.  9.  10. 11. 12. 13.  14.  15.  16.  17.  18.
            x1s = [0, 0, 10, 0, 0, 45, 45, 45, 45, 91, 91, 101, 91, 136, 136]
            x2s = [0, 10, 20, 30, 40, 0, 20, 30, 40, 0, 30, 20, 40, 10, 0]
            ws = [35, 10, 23, 10, 35, 10, 23, 10, 23, 10, 10, 23, 35, 10, 35]
            hs = [10, 10, 10, 10, 10, 20, 10, 10, 10, 20, 10, 10, 10, 40, 10]

            d1 = a
            d2 = a * 50 / Llogo

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
            margin = 1
            d1 = a
            d2 = a
            x1s, x2s, ws, hs = [], [], [], []
            w0 = 4
            h0 = 4
            x10 = np.arange(0, 171, w0)
            x20 = np.arange(0, 171, h0)
            for xi, yi in itertools.product(x10, x20):
                if (xi - 171 / 2) ** 2 + (yi - 171 / 2) ** 2 <= (171 / 2 - w0 / 2) ** 2 * 1.1:
                    x1s.append(xi - w0 / 2)
                    x2s.append(yi - h0 / 2)
                    ws.append(w0)
                    hs.append(h0)
            match case_:
                case "disc":
                    t = np.linspace(0, 6 * gamma, 200)
                case "rasc":
                    t = np.linspace(0, 10 * gamma, 300)
        case _:
            t = np.array([0, ])

    def current_moment(a1, a2, a3):
        moment = 0
        if a3 == 0:
            for xi, yi, wi, hi in zip(x1s, x2s, ws, hs):
                moment += c_j(a1, a2, a3, xi / Llogo * L - d1, yi / Llogo * L - d2, wi / Llogo * L,
                              hi / Llogo * L) / gamma_SI
        return [moment, 0, 0]

    def charge_moment(a1, a2, a3):
        moment = 0
        if a3 == 0:
            for xi, yi, wi, hi in zip(x1s, x2s, ws, hs):
                moment += c_r(a1, a2, a3, xi / Llogo * L - d1, yi / Llogo * L - d2, wi / Llogo * L,
                              hi / Llogo * L) / gamma_SI
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
    gamma_sym = sympy.Symbol("gamma", real=True)
    h_sym = (3*gamma_SI*sympy.sqrt(np.pi/2))**-.5\
        * sympy.exp(-((t_sym-3*gamma)/gamma)**2)*(4*((t_sym-3*gamma)/gamma)**2-2)

    sol = pynoza.solution.Solution(max_order=24,
                                   wave_speed=c0)
    sol.recurse()
    sol.set_moments(current_moment=current_moment,
                    charge_moment=charge_moment)
    match case_:
        case "logo":
            E = sol.compute_e_field(x1, x2, x3, t,
                                    h_sym, t_sym,
                                    verbose=False)
        case "disc":
            E = sol.compute_e_field(x1, x2, x3, t,
                                    h_sym, t_sym,
                                    verbose=False,
                                    delayed=False)

    E1 = E[0, :, :, :, :]

    match case_:
        case "logo":
            fname = "tests/data/Efield_at_2a_and_3a_lambdaOver2_v3.txt"
            data_cmsl = pd.read_csv(fname,
                                    skiprows=range(10),
                                    names=("t", "absE2a", "E2a", "absE3a", "E3a"),
                                    delim_whitespace=True)
            e_cmsl_2a = scipy.interpolate.interp1d(data_cmsl["t"], data_cmsl["E2a"],
                                                   fill_value="extrapolate")(t * gamma_SI)
            e_cmsl_3a = scipy.interpolate.interp1d(data_cmsl["t"], data_cmsl["E3a"],
                                                   fill_value="extrapolate")(t * gamma_SI)
            assert np.linalg.norm(e_cmsl_2a / 2 - E1[0, 0, 0, :], ord=2) / t.size < 160 \
                   and np.linalg.norm(e_cmsl_3a / 2 - E1[0, 0, 1, :], ord=2) / t.size < 170
        case "disc":
            fname = "data/ratio-0-5-v6.txt"
            d = pd.read_csv("tests/data/data_paper.csv",
                            skiprows=1)
            delay = -3*gamma_SI*1e9
            e_paper1 = scipy.interpolate.interp1d(d["X.2"] * Tg * gamma_SI * 1e9 - delay, d["Y.2"],
                                                  fill_value="extrapolate")(t * gamma_SI * 1e9)
            e_paper2 = scipy.interpolate.interp1d(d["X.1"] * Tg * gamma_SI * 1e9 - delay, d["Y.1"],
                                                  fill_value="extrapolate")(t * gamma_SI * 1e9)
            e_paper3 = scipy.interpolate.interp1d(d["X"] * Tg * gamma_SI * 1e9 - delay, d["Y"],
                                                  fill_value="extrapolate")(t * gamma_SI * 1e9)
            assert np.linalg.norm(e_paper1 * 1e2 - E1[0, 0, 0, :] * x3[0] / 1e6, ord=2) / t.size < 0.3 \
                   and np.linalg.norm(e_paper2 * 1e2 - E1[0, 0, 1, :] * x3[1] / 1e6, ord=2) / t.size < 0.3 \
                   and np.linalg.norm(e_paper3 * 1e2 - E1[0, 0, 2, :] * x3[2] / 1e6, ord=2) / t.size < 0.3


if __name__ == "__main__":
    test_solution("logo")
