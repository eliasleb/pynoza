import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import inverse_problem
import itertools
import scipy.interpolate
import time
import pickle


def inverse_problem_hira(**kwargs):
    filename = kwargs.get("filename",
                          "../../../git_ignore/GLOBALEM/hira_v12.txt")
    data = pd.read_csv(filename,
                       delim_whitespace=True, header=8)
    names = ["x", "y", "z"]
    t = 0

    while len(names) < len(data.columns):
        names.append(f"Ex@{t=}")
        names.append(f"Ey@{t=}")
        names.append(f"Ez@{t=}")
        t += 1

    names = names[:len(data.columns)]
    data.set_axis(names, axis=1, inplace=True)

    c0 = 3e8
    f = 500e6 / c0
    gamma = np.sqrt(12 / 7) / f
    t0 = 3 * gamma
    dt = 1e-10  # s
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
    n_times = 1
    while not np.isnan(data.iloc[0, 3 + 3 * n_times]):
        n_times += 1
    dt *= c0
    t = np.arange(0, n_times * dt, dt)

    x1 = data["x"]
    x2 = data["y"]
    x3 = data["z"]
    ex = data.iloc[:, 3:3 * n_times + 3:3]
    ey = data.iloc[:, 4:3 * n_times + 3:3]
    ez = data.iloc[:, 5:3 * n_times + 3:3]

    assert np.all(["Ex" in name for name in ex.columns])
    assert np.all(["Ey" in name for name in ey.columns])
    assert np.all(["Ez" in name for name in ez.columns])
    x1 = np.array(x1)
    x2 = np.array(x2)
    x3 = np.array(x3)
    ex = np.array(ex)
    ey = np.array(ey)
    ez = np.array(ez)
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

        apply_symmetry(-1, 1, 1, -1, 1, 1)
    #    apply_symmetry(1, -1, 1, 1, -1, 1)
    #    apply_symmetry(-1, -1, 1, -1, -1, 1)
        apply_symmetry(1, 1, -1, -1, -1, 1)
        apply_symmetry(-1, 1, -1, 1, -1, 1)
    #    apply_symmetry(1, -1, -1, -1, 1, 1)
    #    apply_symmetry(-1, -1, -1, 1, 1, 1)

    assert i_sym // x1.size == times_added

    x1 = np.concatenate((x1, x1_symmetry))
    x2 = np.concatenate((x2, x2_symmetry))
    x3 = np.concatenate((x3, x3_symmetry))
    ex = np.concatenate((ex, ex_symmetry), axis=0)[:, ::down_sample_time]
    ey = np.concatenate((ey, ey_symmetry), axis=0)[:, ::down_sample_time]
    ez = np.concatenate((ez, ez_symmetry), axis=0)[:, ::down_sample_time]
    t = t[::down_sample_time]

    def to_cartesian(radius, theta, phi):
        return radius*np.cos(phi)*np.sin(theta),  radius*np.sin(phi)*np.sin(theta), radius*np.cos(theta)

    indices_obs = list()
    for ri, ti, pi in itertools.product(obs_r, obs_theta, obs_phi):
        obs_x, obs_y, obs_z = to_cartesian(ri, ti, pi)
        dist = (x1 - obs_x)**2 + (x2 - obs_y)**2 + (x3 - obs_z)**2
        indices_obs.append(dist.argmin())

    x1 = x1[indices_obs]
    x2 = x2[indices_obs]
    x3 = x3[indices_obs]
    ex = ex[indices_obs, :]
    ey = ey[indices_obs, :]
    ez = ez[indices_obs, :]

    r = np.sqrt(x1 ** 2 + x2 ** 2 + x3 ** 2)
    t_max = t.max()

    def force_decay(e, t_delay):
        cut = 0.5 * t_max
        return e * ((t_delay <= cut) + (t_delay > cut)*np.exp(-((t_delay-cut)/gamma)**2))

    td = t.reshape(1, t.size) - r.reshape(r.size, 1)

    ex = force_decay(ex, td)
    ey = force_decay(ey, td)
    ez = force_decay(ez, td)

    print(f"{ex.shape=}")

    e_true = [ex, ey, ez]

    energy = np.sum(ex ** 2 + ey ** 2 + ez ** 2, axis=1)
    energy_max = energy.max(initial=0)

    assert np.all(r > 0)

    plt.ion()
    plt.figure()
    plt.plot(r, energy / energy_max, '.')
    r_min = r.min()
    plt.plot(r, 1 / (r/r_min)**1, '.')
    plt.plot(r, 1 / (r/r_min)**2, '.')
    plt.plot(r, 1 / (r/r_min)**3, '.')

    plt.legend(("data", "1/_r", "1/_r^2", "1/_r^3"))

    plt.show()
    input("[Enter] to continue...")

    def get_h_num(h, t):
        if h.size == 0:
            h_num = np.exp(-((t - t0) / gamma) ** 2) * (4 * ((t - t0) / gamma) ** 2 - 2)
            return h_num
        else:
            tail = 10
            return scipy.interpolate.interp1d(np.linspace(t.min(), t.max(), 1 + h.size + tail),
                                              np.concatenate((np.array((0, )), h.ravel(), np.zeros((tail, )))),
                                              kind="cubic")(t) * np.exp(-1/((t/0.1)**2 + 1e-16))

    estimate = None

    order = int(kwargs.get("order", 1))
    kwargs = {"tol": float(kwargs.get("tol", 1e-3)),
              "n_points": int(kwargs.get("n_points", 20)),
              "error_tol": float(kwargs.get("error_tol", 1E-3)),
              "coeff_derivative": 0,
              "verbose_every": int(kwargs.get("verbose_every", 100)),
              "plot": bool(kwargs.get("plot", True)),
              "scale": float(kwargs.get("scale", 1e4)),
              "h_num": get_h_num,
              "find_center": bool(kwargs.get("find_center", True)),
              "max_global_tries": int(kwargs.get("max_global_tries", 1)),
              "compute_grid": False,
              "estimate": estimate}
    shape_mom = (order + 2, order + 2, order + 2, 3)
    dim_mom = 3 * sum([1 for i, j, k in
                       itertools.product(range(order + 1), range(order + 1), range(order + 1)) if i + j + k <= order])

    def get_current_moment(moment):
        current_moment_ = np.zeros(shape_mom)
        ind = 0
        for a1, a2, a3 in itertools.product(range(order + 1), range(order + 1), range(order + 1)):
            if a1 + a2 + a3 <= order:
                current_moment_[a1, a2, a3, :] = moment[ind:ind + 3]
                ind += 3
        assert ind == moment.size
        return current_moment_

    args = (order + 1, e_true, x1, x2, x3, t, get_current_moment, dim_mom)
    current_moment, h, center, e_opt = inverse_problem.inverse_problem(*args, **kwargs)
    estimate = (current_moment, h, center)

    if kwargs.get("plot", True):
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
        match input("Save? [y/*] "):
            case ("y" | "Y"):
                res = pd.DataFrame(data={"t": x1, "x2": x2, "x3": x3}
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
    parser.add_argument("--verbose_every")
    parser.add_argument("--plot")
    parser.add_argument("--scale")
    parser.add_argument("--find_center")
    parser.add_argument("--max_global_tries")
    parser.add_argument("--filename")

    kwargs = parser.parse_args()
    inverse_problem_hira(**vars(kwargs))
