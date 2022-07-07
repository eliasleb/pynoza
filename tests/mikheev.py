import numpy as np
import pynoza
import pynoza.solution
import matplotlib.pyplot as plt
import itertools
import inverse_problem
import pandas as pd
import scipy.interpolate
import time
import pickle
import scipy.special
import pandas as pd


def cot(x, *args, **kwargs):
    return 1 / np.tan(x, *args, **kwargs)


def focal_point_y_coordinate(f, d):
    return f * (1 - d ** 2 / 16)


def read_paper_data():
    first_part = pd.read_csv("data/data_mikheev_part_1.csv", skiprows=1)
    second_part = pd.read_csv("data/data_mikheev_part_2.csv", skiprows=1)
    points_first_part = [1, 2, 3, 6, 10, 7]
    points_second_part = [4, 5, 8, 9]
    data = {}

    def scroll_through_cols(df, points):
        d = {}
        for i, p in enumerate(points):
            if i == 0:
                name_x = "X"
                name_y = "Y"
            else:
                name_x = f"X.{i + len(points)}"
                name_y = f"Y.{i + len(points)}"
            d[p] = {"x": np.array(df[name_x][np.isfinite(df[name_x])]),
                    "y": np.array(df[name_y][np.isfinite(df[name_y])])}
        return d

    data = data | scroll_through_cols(first_part, points_first_part)
    data = data | scroll_through_cols(second_part, points_second_part)

    return data


def ez_mikheev(x, y, z, t, v, *args):
    f_g, c, d, f = args
    t = t.reshape((1, t.size))
    fpy = focal_point_y_coordinate(f, d)
    r = np.sqrt(x ** 2 + (y - fpy) ** 2 + z ** 2)
    r2 = np.sqrt(x ** 2 + y ** 2 + (z - d / 2) ** 2)
    r2p = np.sqrt(x ** 2 + y ** 2 + (z + d / 2) ** 2)
    length = np.sqrt(fpy ** 2 + (d / 2) ** 2)
    beta = np.arctan2(d / 2, fpy)
    k = 2 * x / d
    p = 2 * z / d

    if np.any(np.abs(k) > 1) or np.any(np.abs(p) > 1):
        raise RuntimeError("k and p must be smaller than 1")

    return 1 / 2 / f_g / np.pi * (v(t - r / c) / 2 / r * ((np.sin(beta) + z / r) / (1 + ((y - cot(beta) * d / 2)
                                                                                         * np.cos(beta)
                                                                                         + z * np.sin(beta)) / r)
                                                          + (np.sin(beta) - z / r) / (1 + ((y - cot(beta) * d / 2)
                                                                                           * np.cos(beta)
                                                                                           - z * np.sin(beta)) / r))
                                  - v(t - length / c - r2 / c) / 2 / r2 * (np.sin(beta) - (z - d / 2) / r2) /
                                  (1 + (y * np.cos(beta) + (d / 2 - z) * np.sin(beta)) / r2)
                                  - v(t - length / c - r2p / c) / 2 / r2p * (np.sin(beta) + (z + d / 2) / r2p) /
                                  (1 + (y * np.cos(beta) + (d / 2 + z) * np.sin(beta)) / r2p)
                                  - 4 * v(t - 2 * f / c - y / c + (d / 2 / c) * cot(beta)) / d
                                  * (1 + k ** 2 - p ** 2) / (
                                          1 + 2 * k ** 2 - 2 * p ** 2 + 2 * k ** 2 * p ** 2 + k ** 4 + p ** 4)
                                  + v(t - length / c - r2 / c) / 2 / r2 * (d / 2 - z) / (r2 - y)
                                  + v(t - length / c - r2p / c) / 2 / r2p * (d / 2 + z) / (r2p - y))


def mikheev(**kwargs):
    plot = kwargs.get("plot").lower() == "true"

    z_f = 500
    c0 = 3e8
    eta0 = 377
    f_g = z_f / eta0
    f_over_d = 0.37
    d = .9
    f = f_over_d * d
    amplitude = 9.9
    rise_time = 80e-12 * c0
    duration = 50e-9 * c0

    def v(t_):
        return (t_ > 0) * (t_ < rise_time) * t_ / rise_time * amplitude \
               + (t_ >= rise_time) * (t_ < rise_time + duration) * amplitude

    x1 = np.array((0, .3, 0, 0, .3, 0, 0, .3, 0, .21,)) * d
    x2 = np.array((.6, .6, .6, 1, 1, 1, 1.5, 1.5, 1.5, 1,)) * d
    x3 = np.array((0, 0, .3, 0, 0, .3, 0, 0, .3, .21,)) * d
    x1 = np.concatenate((x1, -x1, x1))
    x2 = np.concatenate((x2, x2, x2))
    x3 = np.concatenate((x3, x3, -x3))

    assert len(x1) == len(x2) and len(x2) == len(x3)

    #   indices = slice(0, 1)
    #   x1 = x1[indices]
    #   x2 = x2[indices]
    #   x3 = x3[indices]
    t = np.linspace(0, 3.5, 50)

    x1 = np.array(x1).reshape((len(x1), 1))
    x2 = np.array(x2).reshape((len(x1), 1))
    x3 = np.array(x3).reshape((len(x1), 1))
    data_paper = read_paper_data()

    if False:
        ez = ez_mikheev(x1, x2, x3, t, v, f_g, 1, d, f)
    else:
        ez = []
        ez_symmetrical_x = []
        ez_symmetrical_z = []
        thresh = 0.1
        for p, xi, yi, zi in zip(data_paper, x1, x2, x3):
            ri = np.sqrt(xi ** 2 + yi ** 2 + zi ** 2)
            start = np.where(np.abs(data_paper[p]["y"]) > thresh)[0][0]
            ezi = pynoza.solution.Interpolator(data_paper[p]["x"][start:] * 1e-9 * c0,
                                               data_paper[p]["y"][start:])(t - ri)
            ez.append(ezi)
            ez_symmetrical_x.append(ezi)
            ez_symmetrical_z.append(ezi)
        ez = np.array(ez + ez_symmetrical_x + ez_symmetrical_z)

    #  theta = np.linspace(0, np.pi, 5)
    #  phi = np.concatenate((np.linspace(0, 40, 5),
    #                        np.linspace(140, 180, 5))) / 180 * np.pi
    #  for theta_i, phi_i in itertools.product(theta, phi):
    #      ez = np.concatenate((ez, np.zeros((1, t.size))), axis=0)
    #      x1 = np.concatenate((x1, np.array([d * np.sin(theta_i) * np.cos(phi_i)]).reshape((1, 1))), axis=0)
    #      x2 = np.concatenate((x2, np.array([d * np.sin(theta_i) * np.sin(phi_i)]).reshape((1, 1))), axis=0)
    #      x3 = np.concatenate((x3, np.array([d * np.cos(theta_i)]).reshape((1, 1))), axis=0)
    ex = np.zeros(ez.shape)
    ey = np.zeros(ez.shape)
    e_true = np.stack((ex, ey, ez))

    with pynoza.PlotAndWait():
        ax = plt.figure().add_subplot(projection='3d')
        ax.quiver(x1.squeeze(), x2.squeeze(), x3.squeeze(),
                  ex.max(axis=1), ey.max(axis=1), ez.max(axis=1),
                  length=.1, normalize=False)
        indices = np.max(ez, axis=1) == 0
        ax.scatter(x1.squeeze()[indices],
                   x2.squeeze()[indices],
                   x3.squeeze()[indices])


#   if plot:
#       plt.figure()
#       plt.ion()
#       r = np.sqrt(x1 ** 2 + x2 ** 2 + x3 ** 2)
#       for p in data_paper:
#           plt.subplot(5, 2, p)
#           plt.plot(data_paper[p]["x"] * 1e-9 * c0, data_paper[p]["y"])
#           plt.plot(t.squeeze(), ez[p - 1, :])
#           plt.title(f"r={r[p - 1]}")
#       plt.tight_layout()
#       plt.pause(0.001)
#       plt.show()
#       input("[Enter] to continue...")

    tail = int(kwargs["n_tail"])

    def get_h_num(h, t_):
        return scipy.interpolate.interp1d(np.linspace(t_.min(), t_.max(), 1 + h.size + tail),
                                          np.concatenate((np.array((0,)), h.ravel(), np.zeros((tail,)))),
                                          kind="cubic")(t_)

    estimate = None

    order = int(kwargs.get("order", 0))
    kwargs = {"tol": float(kwargs.get("tol", 1e-3)),
              "n_points": int(kwargs.get("n_points", 20)),
              "error_tol": float(kwargs.get("error_tol", 1E-3)),
              "coeff_derivative": 0,
              "verbose_every": int(kwargs.get("verbose_every", 100)),
              "plot": plot,
              "scale": float(kwargs.get("scale", 1e4)),
              "h_num": get_h_num,
              "find_center": bool(kwargs.get("find_center", True)),
              "max_global_tries": 1,
              "compute_grid": False,
              "estimate": estimate,
              "p": int(kwargs.get("norm"))}

    shape_mom = (order + 3, order + 3, order + 3, 3)
    dim_mom = 1 * sum((1 for i, j, k in
                      itertools.product(range(order + 1), range(order + 1), range(order + 1)) if i + j + k <= order))

    def get_current_moment(moment):
        current_moment_ = np.zeros(shape_mom)
        ind = 0
        for a1, a2, a3 in itertools.product(range(order + 1), range(order + 1), range(order + 1)):
            if a1 + a2 + a3 <= order:
                current_moment_[a1, a2, a3, 2] = moment[ind]
                ind += 1
        assert ind == moment.size
        return current_moment_

    args = (order + 2, e_true, x1, x2 - focal_point_y_coordinate(f, d), x3, t, None, get_current_moment, dim_mom)
    current_moment, h, center, e_opt = inverse_problem.inverse_problem(*args, **kwargs)

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
    if kwargs["plot"]:
        answer = input("Save? [y/*] ")
    else:
        answer = "y"
    match answer:
        case ("y" | "Y"):
            res = pd.DataFrame(data={"x1": x1.squeeze(), "x2": x2.squeeze(), "x3": x3.squeeze()}
                                     | {f"ex_opt@t={t[i]}": e_opt[0, :, i] for i in range(ex.shape[1])}
                                     | {f"ey_opt@t={t[i]}": e_opt[1, :, i] for i in range(ey.shape[1])}
                                     | {f"ez_opt@t={t[i]}": e_opt[2, :, i] for i in range(ez.shape[1])}
                                     | {f"ex_true@t={t[i]}": ex[:, i] for i in range(ex.shape[1])}
                                     | {f"ey_true@t={t[i]}": ey[:, i] for i in range(ey.shape[1])}
                                     | {f"ez_true@t={t[i]}": ez[:, i] for i in range(ez.shape[1])})
            filename = f"../../../git_ignore/GLOBALEM/opt-result-{time.asctime()}.csv"
            res.to_csv(path_or_buf=filename)
            with open(filename + "_params.pickle", 'wb') as handle:
                pickle.dump((current_moment, h, center, e_true, e_opt), handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved as '{filename}'.")
        case _:
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--down_sample_time", help="Time down-sampling factor")
    parser.add_argument("--r_obs_min")
    parser.add_argument("--r_obs_max")
    parser.add_argument("--n_r_obs")
    parser.add_argument("--n_theta_obs")
    parser.add_argument("--n_phi_obs")
    parser.add_argument("--phi_obs_min")
    parser.add_argument("--phi_obs_max")
    parser.add_argument("--theta_obs_min")
    parser.add_argument("--theta_obs_max")
    parser.add_argument("--order", required=True)
    parser.add_argument("--tol", required=True)
    parser.add_argument("--n_points", required=True)
    parser.add_argument("--n_tail", required=True)
    parser.add_argument("--verbose_every", required=True)
    parser.add_argument("--plot", required=True)
    parser.add_argument("--scale", required=True)
    parser.add_argument("--find_center", required=True)
    parser.add_argument("--norm", required=True)

    kwargs = parser.parse_args()
    mikheev(**vars(kwargs))
