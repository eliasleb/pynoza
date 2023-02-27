import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import inverse_problem
import itertools
import scipy.interpolate
import time
import pickle
# import mikheev
# import pynoza


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

    center_x = float(kwargs.get("center_x", 0.))

    # mikheev_data = mikheev.read_paper_data()
    # d = .9
    # x1 = np.array((0, .3, 0, 0, .3, 0, 0, .3, 0, .21,)) * d
    # x2 = np.array((.6, .6, .6, 1, 1, 1, 1.5, 1.5, 1.5, 1,)) * d
    # x3 = np.array((0, 0, .3, 0, 0, .3, 0, 0, .3, .21,)) * d
    # ez = []
    # t = np.linspace(-kwargs["before"], 3.5, 500)
    # thresh = 1
    # plt.figure()
    # for p, xi, yi, zi in zip(mikheev_data, x1, x2, x3):
    #     ri = np.sqrt(xi ** 2 + yi ** 2 + zi ** 2)
    #     start = np.where(np.abs(mikheev_data[p]["y"]) > thresh)[0][0]
    #     ezi = pynoza.solution.Interpolator((mikheev_data[p]["x"][start:] - mikheev_data[p]["x"][start]) * 1e-9 * c0,
    #                                        mikheev_data[p]["y"][start:])(t - ri)
    #     ez.append(ezi)
#
    # ez = np.array(ez)
    # ex = np.zeros(ez.shape)
    # ey = np.zeros(ez.shape)
#
    # # SKIP pts 1, 2, 3
    # x1 = x1[3:]
    # x2 = x2[3:]
    # x3 = x3[3:]
    # ex = ex[3:, :]
    # ey = ey[3:, :]
    # ez = ez[3:, :]

    obs_x1 = np.array([float(xi) for xi in kwargs.get("x1", [1., ])])
    obs_x2 = np.array([float(xi) for xi in kwargs.get("x2", [0., ])])
    obs_x3 = np.array([float(xi) for xi in kwargs.get("x3", [0., ])])
    before = kwargs["before"]
    coeff_derivative = kwargs["coeff_derivative"]

    assert len(obs_x1) == len(obs_x2) == len(obs_x3)

    filename = kwargs.get("filename",
                          "../../../git_ignore/GLOBALEM/hira_v12.txt")
    try:
        with open("data/x1x2x3exeyez_at_obs.pickle", "rb") as fd:
            x1, x2, x3, ex, ey, ez = pickle.load(fd)
            t = np.arange(0, ex.shape[1] * dt, dt)
            print("Successfully read file")
    except FileNotFoundError:
        x1, x2, x3, ex, ey, ez = read_comsol_file(filename)

        n_added_samples = int(before / dt)
        t = np.arange(-n_added_samples * dt, ex.shape[1] * dt, dt)

        indices_obs = list()
        for obs_x, obs_y, obs_z in zip(obs_x1, obs_x2, obs_x3):
            dist = (x1 - obs_x) ** 2 + (x2 - obs_y) ** 2 + (x3 - obs_z) ** 2
            indices_obs.append(dist.argmin())

        x1 = x1[indices_obs]
        x2 = x2[indices_obs]
        x3 = x3[indices_obs]

        ex = ex[indices_obs, :]
        ey = ey[indices_obs, :]
        ez = ez[indices_obs, :]
        ex = np.pad(ex, pad_width=((0, 0), (n_added_samples, 0)), mode="constant", constant_values=0)
        ey = np.pad(ey, pad_width=((0, 0), (n_added_samples, 0)), mode="constant", constant_values=0)
        ez = np.pad(ez, pad_width=((0, 0), (n_added_samples, 0)), mode="constant", constant_values=0)

        t = t[::down_sample_time]
        ex = ex[:, ::down_sample_time]
        ey = ey[:, ::down_sample_time]
        ez = ez[:, ::down_sample_time]

        with open("data/x1x2x3exeyez_at_obs.pickle", "wb") as fd:
            pickle.dump((x1, x2, x3, ex, ey, ez), fd)
    print(f"{center_x}")
    x1 = x1 - center_x

    r = np.sqrt(x1 ** 2 + x2 ** 2 + x3 ** 2)
    print(f"{r=}")
    # t_max = np.max(t)

    # td = t.reshape(1, t.size) - r.reshape(r.size, 1)

    # ex = force_decay(ex, td)
    # ey = force_decay(ey, td)
    # ez = force_decay(ez, td)

    # import matplotlib
    # matplotlib.use("TkAgg")
    # import matplotlib.pyplot as plt
    # ax = plt.figure().add_subplot(projection='3d')
    # plt.show(block=False)
    # max_e = max((np.max(np.abs(ex)), np.max(np.abs(ey)), np.max(np.abs(ez))))
    # for ind_t, ti in enumerate(t[::4]):
    #     plt.title(f"{ti=:.3f}")
    #     ax.quiver(x1, x2, x3, ex[:, ind_t]/max_e, ey[:, ind_t]/max_e, ez[:, ind_t]/max_e, length=.2,
    #               normalize=False)
    #     plt.waitforbuttonpress()
    #     ax.clear()

    print(f"{ex.shape=}")

    e_true = [ex, ey, ez]

    assert np.all(r > 0)

    tail = int(kwargs["n_tail"])

    # beta = 0.05
    # tr = 160e-12 * c0
    # erfc = scipy.special.erfc

    def get_h_num(h_, t_):
        if h_.size == 0:
            h_num = np.exp(-((t_ - t0) / gamma) ** 2) * (4 * ((t_ - t0) / gamma) ** 2 - 2)
            return h_num
        elif h_.size == 2:
            h_num = lambda delay: np.exp(-((t_ - t0 - delay) / gamma) ** 2) * (4 * ((t_ - t0 - delay) / gamma) ** 2 - 2)
            return h_num(0) + h_[0] * h_num(h[1] ** 2)
        else:
            return scipy.interpolate.interp1d(np.linspace(np.min(t_), np.max(t_), 1 + h_.size + tail),
                                              np.concatenate((np.array((0,)), h_.ravel(), np.zeros((tail,)))),
                                              kind="cubic")(t_)  # * np.exp(-1/((t/0.1)**2 + 1e-16))

    estimate = None

    order = int(kwargs.get("order", 1))
    kwargs = {"tol": 1e-15,
              "n_points": int(kwargs.get("n_points", 20)),
              "error_tol": 1e-15,
              "coeff_derivative": coeff_derivative,
              "verbose_every": int(kwargs.get("verbose_every", 100)),
              "plot": kwargs.get("plot").lower() == "true",
              "scale": float(kwargs.get("scale", 1e4)),
              "h_num": get_h_num,
              "find_center": kwargs.get("find_center", "true").lower() == "true",
              "max_global_tries": 1,
              "compute_grid": False,
              "estimate": estimate,
              "p": 2
              }
    shape_mom = (3, order + 3, order + 3, order + 3)

    def zero_moments(ax, ay, az):
        dims = set()
        if ax % 2 == 0:
            dims = dims.union({2, 3})
        else:
            dims.add(1)
        if az % 2 == 0:
            dims = dims.union({1, 2})
        else:
            dims.add(3)
        if ay % 2 == 0:
            dims.add(2)
        else:
            dims = dims.union({1, 3})
        #######
        if az > 0:
            return {1, 2, 3}

        return dims

    dim_mom = sum([len({1, 2, 3}.difference(zero_moments(i, j, k)))
                   for i, j, k in itertools.product(range(order + 1), range(order + 1), range(order + 1))
                   if i + j + k <= order])

    def get_current_moment(moment):
        current_moment_ = np.zeros(shape_mom)
        ind = 0
        for a1, a2, a3 in itertools.product(range(order + 1), range(order + 1), range(order + 1)):
            if a1 + a2 + a3 <= order:
                dims = {1, 2, 3}.difference(zero_moments(a1, a2, a3))
                for dim in dims:
                    current_moment_[dim - 1, a1, a2, a3] = moment[ind] / 10**(a1 + a2 + a3)
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
        max_h = np.max(np.abs(h))
        if max_h > 0:
            plt.plot(t, h / max_h)
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
    parser.add_argument("--x1", nargs="+", required=True)
    parser.add_argument("--x2", nargs="+", required=True)
    parser.add_argument("--x3", nargs="+", required=True)
    parser.add_argument("--order")
    parser.add_argument("--n_points")
    parser.add_argument("--n_tail", required=True)
    parser.add_argument("--verbose_every")
    parser.add_argument("--plot")
    parser.add_argument("--scale")
    parser.add_argument("--find_center")
    parser.add_argument("--center_x")
    parser.add_argument("--filename", required=True)
    parser.add_argument("--dt", help="Sampling time, in second", required=True)
    parser.add_argument("--before", required=True, type=float)
    parser.add_argument("--coeff_derivative", required=True, type=float)

    parsed = parser.parse_args()
    inverse_problem_hira(**vars(parsed))
