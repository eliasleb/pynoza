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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import scipy
import itertools
import inverse_problem
import time
import pickle


def import_data(filename):
    with open(filename) as fd:
        for _ in range(8):
            fd.readline()
        header = fd.readline()
    t = np.array([float(s.split(" temw")[0]) for s in re.split(r"@ t=", header.split("Z")[1])[1:]][::6])
    assert np.all(np.diff(t) > 1e-18)

    data = np.array(pd.read_csv(filename,
                                skiprows=range(9),
                                header=None,
                                delim_whitespace=True))
    x1, x2, x3 = data[:, 0], data[:, 1], data[:, 2]
    ex, ey, ez = data[:, 3::6], data[:, 4::6], data[:, 5::6]
    bx, by, bz = data[:, 6::6], data[:, 7::6], data[:, 8::6]
    shape = (x1.size, t.size)
    assert x1.size == x2.size and x2.size == x3.size
    assert ex.shape == shape and ey.shape == shape and ez.shape == shape
    assert bx.shape == shape and by.shape == shape and bz.shape == shape
    return t, x1, x2, x3, ex, ey, ez, bx, by, bz


def add_symmetry(x1, x2, x3, ex, ey, ez, direction: 0 | 1 | 2, symmetry: str, pos_axis=0):
    match direction:
        case 0:
            x1, x2, x3 = np.concatenate((x1, -x1)), np.concatenate((x2, x2)), np.concatenate((x3, x3))
            match symmetry:
                case "PEC":
                    ex, ey, ez = np.concatenate((ex, ex), axis=pos_axis), np.concatenate((ey, -ey), axis=pos_axis), \
                                 np.concatenate((ez, -ez), axis=pos_axis)
                case "PMC":
                    ex, ey, ez = np.concatenate((ex, -ex)), np.concatenate((ey, ey)), np.concatenate((ez, ez))
        case 1:
            x1, x2, x3 = np.concatenate((x1, x1)), np.concatenate((x2, -x2)), np.concatenate((x3, x3))
            match symmetry:
                case "PEC":
                    ex, ey, ez = np.concatenate((ex, -ex)), np.concatenate((ey, ey)), np.concatenate((ez, -ez))
                case "PMC":
                    ex, ey, ez = np.concatenate((ex, ex)), np.concatenate((ey, -ey)), np.concatenate((ez, ez))
        case 2:
            x1, x2, x3 = np.concatenate((x1, x1)), np.concatenate((x2, x2)), np.concatenate((x3, -x3))
            match symmetry:
                case "PEC":
                    ex, ey, ez = np.concatenate((ex, -ex)), np.concatenate((ey, -ey)), np.concatenate((ez, ez))
                case "PMC":
                    ex, ey, ez = np.concatenate((ex, ex)), np.concatenate((ey, ey)), np.concatenate((ez, -ez))

    return x1, x2, x3, ex, ey, ez


def main(filename, n_tail, tol, n_points, verbose_every, plot, scale, order, magnetic=False):
    find_center = False
    tail = n_tail

    t, x1, x2, x3, ex_sim, ey_sim, ez_sim, bx_sim, by_sim, bz_sim = import_data(filename)
    n_t = 10
    t = t[::n_t] * 3e8
    n_x = 5
    r = np.sqrt(x1**2 + x2**2 + x3**2)
    wanted_indices = np.where(r > 0.05)[0].ravel()[::n_x]
    x1 = x1[wanted_indices]
    x2 = x2[wanted_indices]
    x3 = x3[wanted_indices]
    ex_sim = ex_sim[wanted_indices, ::n_t]
    ey_sim = ey_sim[wanted_indices, ::n_t]
    ez_sim = ez_sim[wanted_indices, ::n_t]
    bx_sim = bx_sim[wanted_indices, ::n_t]
    by_sim = by_sim[wanted_indices, ::n_t]
    bz_sim = bz_sim[wanted_indices, ::n_t]

    print(f"{ex_sim.shape=}")
    # # ax = plt.figure().add_subplot(projection='3d')
    # # ind_plot = 20
    # # p_scale = 5e3
    # # ax.quiver(x1, x2, x3, ex_sim[:, ind_plot]/p_scale, ey_sim[:, ind_plot]/p_scale, ez_sim[:, ind_plot]/p_scale, )
    # x1, x2, x3, ex_sim, ey_sim, ez_sim = add_symmetry(
    #     x1, x2, x3, ex_sim, ey_sim, ez_sim, 0, "PMC", pos_axis=0
    # )
    # _, _, _, bx_sim, by_sim, bz_sim = add_symmetry(
    #     x1, x2, x3, bx_sim, by_sim, bz_sim, 0, "PEC", pos_axis=0
    # )
    # # ax = plt.figure().add_subplot(projection='3d')
    # # ax.quiver(x1, x2, x3, ex_sim[:, ind_plot]/p_scale, ey_sim[:, ind_plot]/p_scale, ez_sim[:, ind_plot]/p_scale, )
    # x1, x2, x3, ex_sim, ey_sim, ez_sim = add_symmetry(
    #     x1, x2, x3, ex_sim, ey_sim, ez_sim, 1, "PMC", pos_axis=0
    # )
    # _, _, _, bx_sim, by_sim, bz_sim = add_symmetry(
    #     x1, x2, x3, bx_sim, by_sim, bz_sim, 1, "PEC", pos_axis=0
    # )
    # # ax = plt.figure().add_subplot(projection='3d')
    # # ax.quiver(x1, x2, x3, ex_sim[:, ind_plot]/p_scale, ey_sim[:, ind_plot]/p_scale, ez_sim[:, ind_plot]/p_scale, )
    # x1, x2, x3, ex_sim, ey_sim, ez_sim = add_symmetry(
    #     x1, x2, x3, ex_sim, ey_sim, ez_sim, 2, "PEC", pos_axis=0
    # )
    # _, _, _, bx_sim, by_sim, bz_sim = add_symmetry(
    #     x1, x2, x3, bx_sim, by_sim, bz_sim, 2, "PMC", pos_axis=0
    # )
    # # ax = plt.figure().add_subplot(projection='3d')
    # # ax.quiver(x1, x2, x3, ex_sim[:, ind_plot]/p_scale, ey_sim[:, ind_plot]/p_scale, ez_sim[:, ind_plot]/p_scale, )
    # # plt.show()

    def get_h_num(h, t_):
        return scipy.interpolate.interp1d(np.linspace(t_.min(), t_.max(), 1 + h.size + tail),
                                          np.concatenate((np.array((0,)), h.ravel(), np.zeros((tail,)))),
                                          kind="cubic")(t_)

    estimate = None

    kwargs = {
        "tol": tol,
        "n_points": n_points,
        "error_tol": tol,
        "coeff_derivative": 0,
        "verbose_every":verbose_every,
        "plot": plot,
        "scale": scale,
        "h_num": get_h_num,
        "find_center": find_center,
        "max_global_tries": 1,
        "compute_grid": False,
        "estimate": estimate,
        "p": 2,
        "seed": 1,
        "fit_on_magnetic_field": magnetic,
        "b_true": (bx_sim, by_sim, bz_sim),
        "shift": 0,
        "return_residual_error": True
    }

    shape_mom = (3, order + 3, order + 3, order + 3)

    dim_mom = 3 * sum(1 for i, j, k in itertools.product(range(order + 1), repeat=3) if i + j + k <= order)

    def get_current_moment(moment_):
        current_moment_ = np.zeros(shape_mom)
        ind = 0
        for a1, a2, a3 in itertools.product(range(order + 1), repeat=3):
            if a1 + a2 + a3 <= order:
                current_moment_[:, a1, a2, a3] = moment_[ind:ind + 3]
                ind += 3

        assert ind == moment_.size
        return current_moment_
    e_true = (ex_sim, ey_sim, ez_sim)
    args = (order + 2, e_true, x1, x2, x3, t, None, get_current_moment, dim_mom)
    current_moment, h, center, e_opt, residual = inverse_problem.inverse_problem(*args, **kwargs)

    if plot:
        plt.ion()

        print(f"{center=}")

        inverse_problem.plot_moment(current_moment)

        plt.figure()
        plt.plot(t, h / np.max(np.abs(h)))
        plt.xlabel("Time (relative)")
        plt.ylabel("Amplitude (normalized)")
        plt.title("Current vs time")
        plt.pause(0.1)
        plt.show()

    if plot:
        answer = input("Save? [y/*] ")
    else:
        answer = ""
    match answer:
        case ("y" | "Y"):
            res = pd.DataFrame(data={"x1": x1.squeeze(), "x2": x2.squeeze(), "x3": x3.squeeze()}
                                     | {f"ex_opt@t={t[i]}": e_opt[0, :, i] for i in range(ex_sim.shape[1])}
                                     | {f"ey_opt@t={t[i]}": e_opt[1, :, i] for i in range(ey_sim.shape[1])}
                                     | {f"ez_opt@t={t[i]}": e_opt[2, :, i] for i in range(ez_sim.shape[1])}
                                     | {f"ex_true@t={t[i]}": ex_sim[:, i] for i in range(ex_sim.shape[1])}
                                     | {f"ey_true@t={t[i]}": ey_sim[:, i] for i in range(ey_sim.shape[1])}
                                     | {f"ez_true@t={t[i]}": ez_sim[:, i] for i in range(ez_sim.shape[1])})
            filename = f"../../git_ignore/GLOBALEM/opt-result-{time.asctime()}.csv"
            res.to_csv(path_or_buf=filename)
            with open(filename + "_params.pickle", 'wb') as handle:
                pickle.dump((current_moment, h, center, e_true, e_opt), handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved as '{filename}'.")
        case _:
            pass
    print(f"{residual=}")
    assert residual < .185

    fom = np.abs(np.max(h) * current_moment[2, 0, 0, 0])
    print(f"{fom=}")
    return fom


def test_inverse_problem_on_sim_data():

    _kwargs = {
        "filename": "tests/data/dipole_v11.txt",
        "n_tail": 20,
        "tol":  1e-5,
        "n_points": 40,
        "verbose_every": 100,
        "plot": False,
        "scale": 1e10,
        "order":  0,
    }
    fom_electric = main(magnetic=False, **_kwargs)
    fom_magnetic = main(magnetic=True, **_kwargs)
    assert abs(fom_electric - fom_magnetic) < .2


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", required=True, type=str)
    parser.add_argument("--n_tail", required=True, type=int)
    parser.add_argument("--tol", required=True, type=float)
    parser.add_argument("--n_points", required=True, type=int)
    parser.add_argument("--verbose_every", required=True, type=int)
    parser.add_argument("--plot", required=True, type=bool)
    parser.add_argument("--scale", required=True, type=float)
    parser.add_argument("--order", required=True, type=int)

    kwargs = parser.parse_args()
    main(**vars(kwargs))

