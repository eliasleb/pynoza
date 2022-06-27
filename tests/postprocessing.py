import pickle
import inverse_problem
import matplotlib.pyplot as plt
import numpy as np
import pynoza
import scipy.interpolate


def predicted_excitation(x1, y1, x2):
    x1_, y1_, x2_ = np.fft.fft(x1), np.fft.fft(y1), np.fft.fft(x2)
    y2_ = y1_ / x1_ * x2_
    return np.fft.ifft(y2_)


def postprocessing(**kwargs):
    current_moment, h, center, e_true, e_opt = pickle.load(open(kwargs.get("filename"), "rb"))
    inverse_problem.plot_moment(current_moment)

    t = np.linspace(0, 1, h.size)
    gamma = .105 * .8
    t0 = 3.6 * gamma
    h_true = 2.2 * np.exp(-((t - t0) / gamma)**2) * (4 * ((t - t0) / gamma)**2 - 2)

    plt.figure()
    plt.plot(t, h)
    plt.plot(t, h_true, "--")
    plt.xlabel("Time (1)")
    plt.ylabel("Amplitude (1)")
    plt.legend(("Fitted", "Simulation"))

    plt.figure(figsize=(10, 5))

    scale = 1e5
    for component in range(3):
        plt.subplot(1, 3, component + 1)
        plt.plot(t, e_true[component].T, "g--")
        plt.plot(t, e_opt[component].T * scale, "b-")
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

    x1 = np.array([0, ])
    x2 = np.array([-.2, ])
    x3 = np.array([0, ])
    t_fast = np.linspace(0, 1, 200)
    h = scipy.interpolate.interp1d(t, h)(t_fast)
    h_true = scipy.interpolate.interp1d(t, h_true)(t_fast)
    t = t_fast
    t1 = 0.02
    t2 = .08
    t10 = .05
    h1 = (t >= 0) * (t < t10) * (1 - np.exp(-t/t1)) + (t >= t10) * np.exp(-(t - t10) / t2)
    reg = np.exp(-((t - 0.5) / 0.03)**2)
    h1 = np.convolve(h1, reg, "same")

    plt.figure()
    plt.plot(t, h1)
    h2 = np.real(predicted_excitation(h_true, h, h1))
    h2 = h2 - h2[0]
    e_pred = sol.compute_e_field(x1, x2, x3, t, h1, None, compute_grid=False)

    plt.figure()
    plt.plot(t, e_pred[0].T, "r-")
    plt.plot(t, e_pred[1].T, "g.")
    plt.plot(t, e_pred[2].T, "b--")
    plt.legend(("x", "y", "z"))
    plt.xlabel("Time (1)")
    plt.ylabel("Amplitude (1)")

    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--filename", help="Path of the pickle dump file", required=True)

    kwargs_parsed = parser.parse_args()
    postprocessing(**vars(kwargs_parsed))
