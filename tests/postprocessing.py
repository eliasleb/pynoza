import pickle
import inverse_problem
import matplotlib.pyplot as plt
import numpy as np
import pynoza
import scipy.interpolate
import HIRA


def predicted_excitation(x1, y1, x2):
    x1_, y1_, x2_ = np.fft.fft(x1), np.fft.fft(y1), np.fft.fft(x2)
    y2_ = y1_ / x1_ * x2_
    return np.fft.ifft(y2_)


def postprocessing(**kwargs):

    scale = float(kwargs["scale"])
    down_sample_time = int(kwargs["down_sample_time"])

    current_moment, h, center, e_true, e_opt = pickle.load(open(kwargs.get("filename"), "rb"))
    e_true = np.array(e_true)
    inverse_problem.plot_moment(current_moment)

    print(f"L2 Error: {np.sum((e_true - e_opt)**2) / np.sum(e_true**2)}")

    dt = float(kwargs["dt"]) * 3e8 * down_sample_time
    t = np.linspace(0, dt * h.size, h.size)
    gamma = (12 / 7)**.5 / 500e6 * 3e8
    t0 = 3 * gamma
    h_true = np.exp(-((t - t0) / gamma)**2) * (4 * ((t - t0) / gamma)**2 - 2)

    plt.figure(figsize=(5, 3))
    plt.plot(t, h / np.abs(h).max())
    plt.plot(t, h_true / np.abs(h_true).max(), "--")
    plt.xlabel("Time (1)")
    plt.ylabel("Amplitude (1)")
    plt.legend(("Fitted", "Simulation"))
    plt.tight_layout()

    plt.figure(figsize=(10, 5))

    for component in range(3):
        plt.subplot(1, 3, component + 1)
        plt.plot(t, e_true[component].T, "g--")
        plt.plot(t, e_opt[component].T, "b-")
        plt.xlabel("Time (1)")
        plt.ylabel("Amplitude (1)")
        plt.title(f"{['x', 'y', 'z'][component]}-component")
    plt.tight_layout()

    order = current_moment.shape[0] - 1
    sol = pynoza.Solution(max_order=order)
    sol.recurse()
    charge_moment = inverse_problem.get_charge_moment(current_moment)
    current_moment_lambda = lambda a1, a2, a3: list(current_moment[a1, a2, a3, :])
    charge_moment_lambda = lambda a1, a2, a3: list(charge_moment[a1, a2, a3, :])
    sol.set_moments(current_moment=current_moment_lambda, charge_moment=charge_moment_lambda)

    print(f"{center=}")

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--filename", help="Path of the pickle dump file", required=True)
    parser.add_argument("--dt", help="Time step", required=True)
    parser.add_argument("--scale", required=True)
    parser.add_argument("--down_sample_time", required=True)

    kwargs_parsed = parser.parse_args()
    postprocessing(**vars(kwargs_parsed))
