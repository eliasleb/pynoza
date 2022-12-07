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

import itertools
import pickle

import pandas as pd

import inverse_problem
import matplotlib.pyplot as plt
import numpy as np
import pynoza
import scipy.interpolate
import HIRA
import scipy.special
import matplotlib

# Say, "the default sans-serif font is COMIC SANS"
matplotlib.rcParams['font.sans-serif'] = "Times"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "Times"


def predicted_excitation(x1, y1, x2):
    x1_, y1_, x2_ = np.fft.fft(x1), np.fft.fft(y1), np.fft.fft(x2)
    y2_ = y1_ / x1_ * x2_
    return np.fft.ifft(y2_)


def postprocessing_globalem(*args):
    current_moment, down_sample_time, t, h, center, scale, sol, order = args

    r_min = 1
    down_sample_space = 10

    x1, x2, x3, ex, ey, ez = HIRA.read_comsol_file("../../../git_ignore/GLOBALEM/hira_v12.txt")
    indices = []
    for i, (xi, yi, zi) in enumerate(zip(x1, x2, x3)):
        ri = np.sqrt(xi**2 + yi**2 + zi**2)
        if ri > r_min:
            indices.append(i)
    indices = indices[::down_sample_space]
    e_true = np.stack((ex[indices, ::down_sample_time],
                       ey[indices, ::down_sample_time],
                       ez[indices, ::down_sample_time]))
    x1, x2, x3 = x1[indices], x2[indices], x3[indices]

    t_full = np.linspace(t.min(), t.max(), e_true.shape[-1])
    h_full = scipy.interpolate.interp1d(t, h, kind="cubic")(t_full)

    e_pred = sol.compute_e_field(x1 - center[0], x2 - center[1], x3 - center[2], t_full, h_full, None,
                                 compute_grid=False) * scale
    p = 2
    true_energy = np.sum(e_true**p)
    error = np.sum((e_true - e_pred)**p) / true_energy
    print(f"{error=}")

    plt.figure()

    skip = 100
    for comp in range(3):
        plt.subplot(1, 3, comp + 1)
        plt.plot(t_full, e_true[comp, ::skip, :].T, "--")
        plt.plot(t_full, e_pred[comp, ::skip, :].T)

    plt.show()

    def gradient(x, order):
        if order == 0:
            return x
        return np.gradient(gradient(x, order - 1))

    plt.figure()
    for diff_order in range(order + 3):
        y = gradient(h_full, diff_order)
        plt.plot(t_full, y / y.max())

    plt.show()


def postprocessing_mikheev(*args):
    current_moment, t, h, h_true, center, sol, order, filename = args

    with pynoza.PlotAndWait():
        r_obs = [1, 1.5]
        theta_obs = np.linspace(0, np.pi, 5)
        phi_obs = np.linspace(np.pi, 2 * np.pi, 8)
        ax = plt.figure().add_subplot(projection='3d')
        for ri in r_obs:
            x_obs, y_obs, z_obs = [], [], []
            for theta_i, phi_i in itertools.product(theta_obs, phi_obs):
                x_obs.append(ri * np.sin(theta_i) * np.cos(phi_i))
                y_obs.append(ri * np.sin(theta_i) * np.sin(phi_i))
                z_obs.append(ri * np.cos(theta_i))
                if ri == r_obs[0]:
                    ax.scatter(x_obs, y_obs, z_obs, marker="*", c="red")
                else:
                    ax.scatter(x_obs, y_obs, z_obs, marker="d", c="k")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        r_max = np.max(r_obs)
        ax.set_xlim(-r_max, r_max)
        ax.set_ylim(-r_max, r_max)
        ax.set_zlim(-r_max, r_max)
        ax.view_init(elev=22, azim=-60)

        plt.tight_layout()



    n = 8
    l = 1
    X, Y, Z = np.meshgrid(np.linspace(-l, l, n),
                          np.linspace(-l, l, n),
                          np.linspace(-l, l, n),)
    mask = (np.sqrt(X**2 + Y**2 + Z**2) < 1) * (Y < 0)
    U, V, W = 0, Z, -1.65 * Y
    with pynoza.PlotAndWait():
        ax = plt.figure().add_subplot(projection='3d')

        ax.quiver(X, Y, Z, U * mask, V * mask, W * mask, length=0.25, color="k")

        ax.set_xlim((-l, l))
        ax.set_ylim((-l, l))
        ax.set_zlim((-l, l))

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        plt.tight_layout()

    for index in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)):
        print(f"{index=}, {np.sum(U * X**index[0] * Y**index[1] * Z**index[2]):.1f}, \
        {np.sum(V * X**index[0] * Y**index[1] * Z**index[2]):.1f}, \
        {np.sum(W * X**index[0] * Y**index[1] * Z**index[2]):.1f}")

    sol.compute_e_field(np.array([1, ]), np.array([1, ]), np.array([1, ]), np.linspace(0, 1, 100),
                        np.zeros((100, )), None, compute_grid=False, compute_txt=True)
    t0 = 3
    gamma = 1
    term = ((t - t0) / gamma)**2
    h_true = np.exp(-term) * (4 * term - 2)

    with pynoza.PlotAndWait():
        cwidth = 5
        plt.figure(figsize=(cwidth, cwidth * .6))
        max_h = np.max(np.abs(h))
        max_h_true = np.max(np.abs(h_true))
        plt.plot(t, h / max_h, "k")
        plt.plot(t, h_true / max_h_true, "r--")

        plt.xlabel("Time (s c$_0$)")
        plt.ylabel("Amplitude (1)")
        #plt.xlim((0, 7))

        plt.legend(("Equivalent time-\ndependent excitation", "Lumped port voltage"),
                   loc="lower right")
        plt.tight_layout()

    print(to_mathematica(sol.get_e_field_text()))
    if center is None:
        center = [0, 0, 0]

    filename_csv = filename.split("_params.pickle")[0]
    data = pd.read_csv(filename_csv)
    x1, x2, x3 = data["x1"], data["x2"], data["x3"]
    x1, x2, x3 = np.array(x1), np.array(x2), np.array(x3)


    n_added = 0
    r = 2 * x2.max()
    #t = np.concatenate((t, np.linspace(t[-1], t[-1] + n_added * t[-1], int(n_added * t.size))))
    #h = np.concatenate((h, np.zeros(t.size - h.size)))
    import synthetic
    synthetic.plot_directivity(sol, r, h_true, t, phi_min=3.2, phi_max=6.1)

    print(f"{current_moment[2, 0, 1, 0] / current_moment[1, 0, 0, 1]=}")

    with pynoza.PlotAndWait():
        plt.figure(figsize=(cwidth, cwidth * .7))
        ax = plt.gca()
        # 0 => opt-result-Wed\ Aug\ 24\ 14:57:23\ 2022.csv_params.pickle
        # 1 => opt-result-Wed\ Aug\ 24\ 14:48:51\ 2022.csv_params.pickle
        # 2 => opt-result-Wed\ Aug\ 24\ 14:51:32\ 2022.csv_params.pickle
        # 3 => opt-result-Wed\ Aug\ 24\ 12:35:02\ 2022.csv_params.pickle
        orders = [2, 3, 4, 5]
        errors = [0.756, 0.151, 0.140, 0.1004297861797407]
        dof = [19, 23, 31, 45]
        centers = [
            [-0.00050089, -0.11762098, -0.48088337],
            [1.42365544e-04, 1.92644617e-01, 4.37729759e-04],
            [0.00034381,  0.20340903, -0.22510711],
            [-0.00021556,  0.21152769,  0.00137421],
            [-0.12295806, 0.22205782, 0.09210685],
        ]
        plt.plot(orders, np.array(errors) * 100, "k-d")
        plt.xticks(orders, [str(order_i) for order_i in orders])
        plt.xlabel("Truncation order")
        plt.ylabel("Residual error (%)")

        ax2 = ax.twinx()
        ax2.plot(orders, dof, "r-")
        ax2.scatter(orders, dof, c="red", marker="*", s=100)
        ax2.tick_params(axis='y', colors='red')
        ax2.set_ylabel("Degrees of freedom", color="red", )

        plt.tight_layout()


def postprocessing(**kwargs):

    scale = float(kwargs["scale"])
    down_sample_time = int(kwargs["down_sample_time"])

    current_moment, h, center, e_true, e_opt = pickle.load(open(kwargs.get("filename"), "rb"))
    e_true = np.array(e_true)
    inverse_problem.plot_moment(current_moment)

    print(f"The order is {current_moment.shape[1] - 1}")

    print(f"L2 Error: {np.sum((e_true - e_opt)**2) / np.sum(e_true**2)}")

    dt = float(kwargs["dt"]) * 3e8 * down_sample_time
    t = np.linspace(0, dt * h.size, h.size)
    gamma = (12 / 7)**.5 / 500e6 * 3e8
    t0 = 3 * gamma
    h_true = np.exp(-((t - t0) / gamma)**2) * (4 * ((t - t0) / gamma)**2 - 2)

    with pynoza.PlotAndWait(wait_for_enter_keypress=True, figsize=(5, 3), new_figure=True):
        plt.plot(t, h / np.abs(h).max())
        #plt.plot(t, h_true / np.abs(h_true).max(), "--")
        plt.xlabel("Time (1)")
        plt.ylabel("Amplitude (1)")
        #plt.legend(("Fitted", "Simulation"))
        plt.tight_layout()

        plt.figure(figsize=(10, 5))
        e_max = np.abs(e_true).max() * 1.1
        for component in range(3):
            plt.subplot(1, 3, component + 1)
            plt.plot(t, e_true[component].T, "g--")
            plt.plot(t, e_opt[component].T, "b-")
            plt.ylim((-e_max, e_max))
            plt.xlabel("Time (1)")
            plt.ylabel("Amplitude (1)")
            plt.title(f"{['x', 'y', 'z'][component]}-component")
        plt.tight_layout()
        plt.pause(0.0001)
        plt.show()

        print(f"{center=}")

    order = current_moment.shape[1] - 1
    sol = pynoza.Solution(max_order=order)
    sol.recurse()
    charge_moment = pynoza.get_charge_moment(current_moment)
    current_moment_lambda = lambda a1, a2, a3: list(current_moment[:, a1, a2, a3])
    charge_moment_lambda = lambda a1, a2, a3: list(charge_moment[:, a1, a2, a3])
    sol.set_moments(current_moment=current_moment_lambda, charge_moment=charge_moment_lambda)

    match kwargs["case"]:
        case "globalem":
            postprocessing_globalem(current_moment, down_sample_time, t, h, center, scale, sol, order)
        case "mikheev":
            postprocessing_mikheev(current_moment, t, h, h_true, center, sol, order, kwargs.get("filename"))
        case _:
            raise ValueError(f"Unknown --case: `{kwargs['case']}`")


def to_mathematica(expr: str):
    expr = expr.replace("%", "").replace("'", "").replace("[", "{").replace("]", "}").replace("e+", "*^")\
        .replace("e-", "*^-").replace("\n","")
    for order in range(100):
        expr = expr.replace(f"dh_dt^({order})(t-r/1.0)", f"D[H[t-r/c], {{t, {order+1}}}]")
        expr = expr.replace(f"int_h^({order})(t-r/1.0)", f"D[H[t-r/c], {{t, {order}}}]")

    return expr


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--filename", help="Path of the pickle dump file", required=True)
    parser.add_argument("--dt", help="Time step", required=True)
    parser.add_argument("--scale", required=True)
    parser.add_argument("--down_sample_time", required=True)
    parser.add_argument("--case", required=True)

    kwargs_parsed = parser.parse_args()
    postprocessing(**vars(kwargs_parsed))
