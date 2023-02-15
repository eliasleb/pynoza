import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import inverse_problem
import itertools
import scipy.interpolate
import time
import pickle


def read_comsol_file(filename):
    data = pd.read_csv(filename, delim_whitespace=True, header=8)
    names = ["x", "y", "z"]
    t = 0

    while len(names) < len(data.columns):
        names.append(f"Ex@{t=}")
        names.append(f"Ey@{t=}")
        names.append(f"Ez@{t=}")
        t += 1

    names = names[:len(data.columns)]
    data.set_axis(names, axis=1, inplace=True)
    n_times = 1
    while not np.isnan(data.iloc[0, 3 + 3 * n_times]):
        n_times += 1

    x1 = data["x"]
    x2 = data["y"]
    x3 = data["z"]
    ex = data.iloc[:, 3:3 * n_times + 3:3]
    ey = data.iloc[:, 4:3 * n_times + 3:3]
    ez = data.iloc[:, 5:3 * n_times + 3:3]

    del data

    assert np.all(["Ex" in name for name in ex.columns])
    assert np.all(["Ey" in name for name in ey.columns])
    assert np.all(["Ez" in name for name in ez.columns])

    return np.array(x1), np.array(x2), np.array(x3), np.array(ex), np.array(ey), np.array(ez)


def add_symmetries(*args):
    x1, x2, x3, ex, ey, ez = args
    times_added = 3
    x1_symmetry = np.zeros((x1.size * times_added))
    x2_symmetry = np.zeros((x1.size * times_added))
    x3_symmetry = np.zeros((x1.size * times_added))
    ex_symmetry = np.zeros((x1.size * times_added, ex.shape[1]))
    ey_symmetry = np.zeros((x1.size * times_added, ex.shape[1]))
    ez_symmetry = np.zeros((x1.size * times_added, ex.shape[1]))
    i_sym = 0

    for i, args in enumerate(zip(x1, x2, x3, ex, ey, ez)):
        def apply_symmetry(*signature):
            nonlocal x1_symmetry, x2_symmetry, x3_symmetry, ex_symmetry, ey_symmetry, ez_symmetry, i_sym
            x1_symmetry[i_sym], x2_symmetry[i_sym], x3_symmetry[i_sym], \
                ex_symmetry[i_sym, :], ey_symmetry[i_sym, :], ez_symmetry[i_sym, :] \
                = (arg * s for arg, s in zip(args, signature))
            i_sym += 1

        # apply_symmetry(-1, 1, 1, -1, 1, 1)
        apply_symmetry(1, -1, 1, 1, -1, 1)
        # apply_symmetry(-1, -1, 1, -1, -1, 1)
        apply_symmetry(1, 1, -1, -1, -1, 1)
        # apply_symmetry(-1, 1, -1, 1, -1, 1)
        apply_symmetry(1, -1, -1, -1, 1, 1)
        # apply_symmetry(-1, -1, -1, 1, 1, 1)

    assert i_sym // x1.size == times_added

    return np.concatenate((x1, x1_symmetry)), \
           np.concatenate((x2, x2_symmetry)), \
           np.concatenate((x3, x3_symmetry)), \
           np.concatenate((ex, ex_symmetry), axis=0), \
           np.concatenate((ey, ey_symmetry), axis=0), \
           np.concatenate((ez, ez_symmetry), axis=0)


def inverse_problem_hira(**kwargs):

    print(f"{kwargs=}")
    c0 = 3e8
    f = 500e6 / c0
    gamma = np.sqrt(12 / 7) / f
    t0 = 3 * gamma
    dt = float(kwargs["dt"])
    dt *= c0
    down_sample_time = int(kwargs.get("down_sample_time", 6))
    obs_r = np.linspace(float(kwargs.get("r_obs_min", 1)),
                        float(kwargs.get("r_obs_max", 1.6)),
                        int(kwargs.get("n_r_obs", 2)))
    obs_theta = np.linspace(float(kwargs.get("theta_obs_min", 0.)),
                            float(kwargs.get("theta_obs_max", np.pi)),
                            int(kwargs.get("n_theta_obs", 3)))
    obs_phi = np.linspace(float(kwargs.get("phi_obs_min", 0.)),
                          float(kwargs.get("phi_obs_max", 2 * np.pi)),
                          int(kwargs.get("n_phi_obs", 8)))

    center_x = float(kwargs.get("center_x", 0.))

    filename = kwargs.get("filename",
                          "../../../git_ignore/GLOBALEM/hira_v12.txt")
    try:
        with open("data/x1x2x3exeyez_at_obs.pickle", "rb") as fd:
            x1, x2, x3, ex, ey, ez = pickle.load(fd)
            t = np.arange(0, ex.shape[1] * dt, dt)
            print("Successfully read file")
    except FileNotFoundError:
        x1, x2, x3, ex, ey, ez = read_comsol_file(filename)
        x1, x2, x3, ex, ey, ez = add_symmetries(x1, x2, x3, ex, ey, ez)

        ex = ex[:, ::down_sample_time]
        ey = ey[:, ::down_sample_time]
        ez = ez[:, ::down_sample_time]
        t = np.arange(0, ex.shape[1] * dt, dt)
        t = t[::down_sample_time]

        def to_cartesian(radius, theta, phi):
            return radius * np.cos(phi) * np.sin(theta), radius * np.sin(phi) * np.sin(theta), radius * np.cos(theta)

        indices_obs = list()
        for ri, ti, pi in itertools.product(obs_r, obs_theta, obs_phi):
            obs_x, obs_y, obs_z = to_cartesian(ri, ti, pi)
            dist = (x1 - obs_x) ** 2 + (x2 - obs_y) ** 2 + (x3 - obs_z) ** 2
            indices_obs.append(dist.argmin())

        x1 = x1[indices_obs]
        x2 = x2[indices_obs]
        x3 = x3[indices_obs]
        ex = ex[indices_obs, :]
        ey = ey[indices_obs, :]
        ez = ez[indices_obs, :]

        with open("data/x1x2x3exeyez_at_obs.pickle", "wb") as fd:
            pickle.dump((x1, x2, x3, ex, ey, ez), fd)
    print(f"{center_x}")
    x1 = x1 - center_x

    r = np.sqrt(x1 ** 2 + x2 ** 2 + x3 ** 2)
    print(f"{r=}, {x1=}, {x2=}, {x3=}")
    t_max = np.max(t)

    def force_decay(e, t_delay):
        cut = 0.5 * t_max
        return e * ((t_delay <= cut) + (t_delay > cut) * np.exp(-((t_delay - cut) / gamma) ** 2))
    td = t.reshape(1, t.size) - r.reshape(r.size, 1)

    # ex = force_decay(ex, td)
    # ey = force_decay(ey, td)
    # ez = force_decay(ez, td)

    # import matplotlib
    # matplotlib.use("TkAgg")
    # import matplotlib.pyplot as plt
#
    # ax = plt.figure().add_subplot(projection='3d')
    # plt.show(block=False)
    # max_e = max((np.max(np.abs(ex)), np.max(np.abs(ey)), np.max(np.abs(ez))))
    # for ind_t, ti in enumerate(t[::2]):
    #     plt.title(f"{ti=:.3f}")
    #     ax.quiver(x1, x2, x3, ex[:, ind_t]/max_e, ey[:, ind_t]/max_e, ez[:, ind_t]/max_e, length=.2,
    #               normalize=False)
    #     plt.waitforbuttonpress()
    #     ax.clear()

    print(f"{ex.shape=}")

    e_true = [ex, ey, ez]

    assert np.all(r > 0)

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
              "find_center": kwargs.get("find_center", "true").lower() == "true",
              "max_global_tries": int(kwargs.get("max_global_tries", 1)),
              "compute_grid": False,
              "estimate": estimate}
    shape_mom = (3, order + 3, order + 3, order + 3)
    dim_mom = sum([1 for i, j, k in itertools.product(range(order + 1), range(order + 1), range(order + 1))
                       if i + j + k <= order and i % 2 == 1 and k % 2 == 1]) \
    + sum([1 for i, j, k in itertools.product(range(order + 1), range(order + 1), range(order + 1))
                       if i + j + k <= order and i % 2 == 1 and k % 2 == 0]) \
    + sum([1 for i, j, k in itertools.product(range(order + 1), range(order + 1), range(order + 1))
                       if i + j + k <= order and i % 2 == 0 and k % 2 == 1])

    def get_current_moment(moment):
        current_moment_ = np.zeros(shape_mom)
        ind = 0
        for a1, a2, a3 in itertools.product(range(order + 1), range(order + 1), range(order + 1)):
            if a1 + a2 + a3 <= order:
                if a1 % 2 == 1:  # alpha_x is odd
                    if a3 % 2 == 1:  # alpha_z is odd:
                        current_moment_[1, a1, a2, a3] = moment[ind]
                        ind += 1
                    else:
                        current_moment_[2, a1, a2, a3] = moment[ind]
                        ind += 1
                else:  # alpha_x is even
                    if a3 % 2 == 1:
                        current_moment_[0, a1, a2, a3] = moment[ind]
                        ind += 1
        assert ind == moment.size
        return current_moment_

    args = (order + 2, e_true, x1, x2, x3, t, None, get_current_moment, dim_mom)
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
    parser.add_argument("--order")
    parser.add_argument("--tol")
    parser.add_argument("--n_points")
    parser.add_argument("--n_tail", required=True)
    parser.add_argument("--verbose_every")
    parser.add_argument("--plot")
    parser.add_argument("--scale")
    parser.add_argument("--find_center")
    parser.add_argument("--max_global_tries")
    parser.add_argument("--center_x")
    parser.add_argument("--filename", required=True)
    parser.add_argument("--dt", help="Sampling time, in second", required=True)

    parsed = parser.parse_args()
    inverse_problem_hira(**vars(parsed))
