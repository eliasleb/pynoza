import numpy as np
import pynoza
import matplotlib.pyplot as plt
import itertools
import inverse_problem
import pandas as pd
import scipy.interpolate
import time
import pickle


def cot(*args, **kwargs):
    return 1 / np.tan(*args, **kwargs)


def focal_point_y_coordinate(f, d):
    return f * (1 - d ** 2 / 16)


def ez_mikheev(x, y, z, t, v, *args):
    f_g, c, d, f = args
    t = t.reshape((1, t.size))
    fpy = focal_point_y_coordinate(f, d)
    r = np.sqrt(x ** 2 + (y - fpy) ** 2 + z ** 2)
    r2 = np.sqrt(x ** 2 + y ** 2 + (z - d / 2) ** 2)
    r2p = np.sqrt(x ** 2 + y ** 2 + (z + d / 2) ** 2)
    l = np.sqrt(fpy ** 2 + (d / 2) ** 2)
    beta = np.arctan2(d / 2, f)
    k = 2 * x / d
    p = 2 * z / d

    if np.any(np.abs(k) > 1) or np.any(np.abs(p) > 1):
        raise RuntimeError("k and p must be smaller than 1")

    return 1 / 2 / f_g / np.pi * (v(t - r / c) / 2 / r * ((np.sin(beta) + z / r) / (1 + ((y - cot(beta) * d / 2)
                                                                                         * np.cos(beta)
                                                                                         + z * np.sin(beta)) / r)
                                                          + (np.sin(beta) - z / r) / (1 + ((y - cot(beta) * d / 2)
                                                                                           * np.cos(beta)
                                                                                           - z * np.sin(beta))) / r)
                                  - v(t - l / c - r2 / c) / 2 / r2 * (np.sin(beta) - (z - d / 2) / r2) /
                                  (1 + (y * np.cos(beta) + (d / 2 - z) * np.sin(beta)) / r2)
                                  - v(t - l / c - r2p / c) / 2 / r2p * (np.sin(beta) + (z + d / 2) / r2p) /
                                  (1 + (y * np.cos(beta) + (d / 2 + z) * np.sin(beta)) / r2p)
                                  - 4 * v(t - 2 * f / c - y / c + (d / 2 / c) * cot(beta)) / d
                                  * (1 + k ** 2 - p ** 2) / (
                                              1 + 2 * k ** 2 - 2 * p ** 2 + 2 * k ** 2 * p ** 2 + k ** 4 + p ** 4)
                                  + v(t - l / c - r2 / c) / 2 / r2 * (d / 2 - z) / (r2 - y)
                                  + v(t - l / c - r2p / c) / 2 / r2p * (d / 2 + z) / (r2p - y))


def mikheev(**kwargs):
    z_f = 50
    c0 = 3e8
    eta0 = 377
    f_g = z_f / eta0
    f = 500e6 / c0
    gamma = np.sqrt(12 / 7) / f
    t0 = 3 * gamma

    f = .25
    d = 1

    def v(t_):
        return np.exp(-((t_ - t0) / gamma) ** 2) * (4 * ((t_ - t0) / gamma) ** 2 - 2)

    xg = np.linspace(-d / 3, d / 3, 3)
    yg = np.linspace(10 * d, 20 * d, 3)
    zg = xg.copy()

    t = np.arange(0, 6 * gamma + yg.max(), 1 / f / 100)

    x1, x2, x3 = [], [], []
    for xi, yi, zi in itertools.product(xg, yg, zg):
        x1.append(xi)
        x2.append(yi)
        x3.append(zi)

#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection="3d")
#    ax.scatter(x1, x2, x3)
#    ax.set_xlim(min(x1), max(x1))
#    ax.set_ylim(-f, max(x2))
#    ax.set_zlim(min(x3), max(x3))
#    plt.show()

    x1 = np.array(x1).reshape((len(x1), 1))
    x2 = np.array(x2).reshape((len(x1), 1))
    x3 = np.array(x3).reshape((len(x1), 1))

    ez = ez_mikheev(x1, x2, x3, t, v, f_g, 1, d, f)
    ex = np.zeros(ez.shape)
    ey = np.zeros(ez.shape)

    e_true = np.stack((ex, ey, ez))

    tail = int(kwargs["n_tail"])

    def get_h_num(h, t):
        if h.size == 0:
            h_num = np.exp(-((t - t0) / gamma) ** 2) * (4 * ((t - t0) / gamma) ** 2 - 2)
            return h_num
        elif h.size == 2:
            h_num = lambda delay: np.exp(-((t - t0 - delay) / gamma) ** 2) * (4 * ((t - t0 - delay) / gamma) ** 2 - 2)
            return h_num(0) + h[0] * h_num(h[1] ** 2)
        else:
            return scipy.interpolate.interp1d(np.linspace(t.min(), t.max(), 1 + h.size + tail),
                                              np.concatenate((np.array((0,)), h.ravel(), np.zeros((tail,)))),
                                              kind="cubic")(t)  # * np.exp(-1/((t/0.1)**2 + 1e-16))

    estimate = None

    order = int(kwargs.get("order", 1))
    kwargs = {"tol": float(kwargs.get("tol", 1e-3)),
              "n_points": int(kwargs.get("n_points", 20)),
              "error_tol": float(kwargs.get("error_tol", 1E-3)),
              "coeff_derivative": 0,
              "verbose_every": int(kwargs.get("verbose_every", 100)),
              "plot": kwargs.get("plot").lower() == "true",
              "scale": float(kwargs.get("scale", 1e4)),
              "h_num": get_h_num,
              "find_center": bool(kwargs.get("find_center", True)),
              "max_global_tries": 1,
              "compute_grid": False,
              "estimate": estimate,
              "p": int(kwargs.get("norm"))}

    shape_mom = (order + 2, order + 2, order + 2, 3)
    dim_mom = sum([1 for i, j, k in
                   itertools.product(range(order + 1), range(order + 1), range(order + 1)) if i + j + k <= order])

    def get_current_moment(moment):
        current_moment_ = np.zeros(shape_mom)
        ind = 0
        for a1, a2, a3 in itertools.product(range(order + 1), range(order + 1), range(order + 1)):
            if a1 + a2 + a3 <= order:
                current_moment_[a1, a2, a3, 2] = moment[ind]
                ind += 1
        assert ind == moment.size
        return current_moment_

    args = (order + 1, e_true, x1, x2, x3, t, get_current_moment, dim_mom)
    current_moment, h, center, e_opt = inverse_problem.inverse_problem(*args, **kwargs)

    if kwargs["plot"]:
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