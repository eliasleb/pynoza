import numpy as np
import pynoza
import matplotlib.pyplot as plt
import itertools
import inverse_problem


def build_current_moment(order):
    current_moment = np.zeros((order + 3, order + 3, order + 3, 3))
    for i in range(order + 1):
        current_moment[0, 0, i, 2] = (-1) ** i / (i + 1)
    return current_moment


def plot_directivity(solution, r, h, t):
    theta = np.linspace(0, np.pi, 30)
    phi = np.linspace(0, 2 * np.pi, 60)
    coords_directivity = [[], [], []]
    for theta_i, phi_i in itertools.product(theta, phi):
        coords_directivity[0].append(r * np.sin(theta_i) * np.cos(phi_i))
        coords_directivity[1].append(r * np.sin(theta_i) * np.sin(phi_i))
        coords_directivity[2].append(r * np.cos(theta_i))

    coords_directivity = np.array(coords_directivity)
    e_pred = solution.compute_e_field(coords_directivity[0], coords_directivity[1], coords_directivity[2],
                                      t, h, None, compute_grid=False)

    with pynoza.PlotAndWait(new_figure=True):
        plt.plot(t, e_pred[2, :, :].T)

    energy = np.sum(e_pred**2, axis=(0, 2))
    energy = energy / np.max(energy)

    with pynoza.PlotAndWait(new_figure=False):
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(coords_directivity[0] / r * energy,
                   coords_directivity[1] / r * energy,
                   coords_directivity[2] / r * energy, color="b")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")


def synthetic_antenna():
    order = 11
    current_moment = build_current_moment(order)
    charge_moment = inverse_problem.get_charge_moment(current_moment)
    current_moment_callable = lambda a1, a2, a3: list(current_moment[a1, a2, a3, :])
    charge_moment_callable = lambda a1, a2, a3: list(charge_moment[a1, a2, a3, :])

    sol = pynoza.Solution(max_order=order + 2)
    sol.recurse()
    sol.set_moments(current_moment=current_moment_callable,
                    charge_moment=charge_moment_callable)
    t = np.linspace(0, 5, 100)
    f = 4
    gamma = np.sqrt(12 / 7) / f
    t0 = 3 * gamma
    h = np.exp(-((t - t0) / gamma)**2) * (4 * ((t - t0) / gamma)**2 - 2)
    r = 2.5
    plot_directivity(sol, r, h, t)


if __name__ == "__main__":
    synthetic_antenna()