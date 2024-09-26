import numpy as np
import matplotlib.pyplot as plt
from pynoza import Solution
from itertools import product
from pynoza.from_mathematica import cartesian_from_real_spherical
from pynoza.helpers import get_poynting
from scipy.special import lpmv


def get_moment(*args, lm=(0, 0), pol=2):
    d = cartesian_from_real_spherical(*lm)
    coefficient = d.get(args, 0.)
    assert np.isreal(coefficient)
    result = [0., 0., 0.]
    result[pol] = coefficient
    return result


def evaluate_radiation_patter(l, m, pol=2, formula=None, plot=False):
    c0_si = 3e8
    f0 = 1e9 / c0_si
    t = np.linspace(0, 50e-9, 1_000) * c0_si
    t0 = 70e-9 * c0_si
    sigma = 20e-9 * c0_si
    h = np.sin(2 * np.pi * f0 * t) * np.exp(-1 / 2 * ((t - t0) / sigma)**2)

    # plt.figure()
    # plt.plot(t, h)

    sol = Solution(max_order=l+2)
    sol.recurse()
    sol.set_moments(current_moment=lambda *args: get_moment(*args, lm=(l, m), pol=pol))

    r = 5
    d_angle = 90 / 10 * np.pi / 180
    theta, phi = np.arange(0, np.pi, d_angle), np.arange(0, 2 * np.pi, d_angle)
    x, y, z = [], [], []
    for theta_i, phi_i in product(theta, phi):
        x.append(r * np.cos(phi_i) * np.sin(theta_i))
        y.append(r * np.sin(phi_i) * np.sin(theta_i))
        z.append(r * np.cos(theta_i))
    x, y, z = np.array(x), np.array(y), np.array(z)
    e_field = sol.compute_e_field(x, y, z, t, h, None, compute_grid=False)
    b_field = sol.compute_b_field(x, y, z, t, h, None, compute_grid=False)
    poynting = get_poynting(e_field, b_field / (4 * np.pi * 1e-7))
    directivity = np.mean(poynting**2, axis=(0, -1,)).reshape((theta.size, phi.size))**.5

    if plot:
        if formula is not None:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
        plt.contourf(phi*180/np.pi, theta*180/np.pi, directivity,
                     cmap="jet", levels=np.linspace(0, np.max(directivity), 21))
        plt.title(f"{l}, {m}")

    if formula is not None:
        if plot:
            plt.subplot(1, 2, 2)
        analytic = formula(theta[:, None], phi[None, :])
        if analytic.shape == (theta.size, 1):
            analytic = np.repeat(analytic, phi.size, 1)
        elif analytic.shape == (1, phi.size):
            analytic = analytic[None, :]

        if plot:
            plt.contourf(phi*180/np.pi, theta*180/np.pi, analytic,
                         cmap="jet", levels=np.linspace(0, np.max(analytic), 21))
    if formula is not None:
        directivity /= np.max(directivity)
        analytic /= np.max(analytic)
        assert np.allclose(analytic, directivity, atol=5e-4)


def test(plot=False):
    evaluate_radiation_patter(0, 0, formula=lambda t, p: np.sin(t) ** 2, plot=plot)
    evaluate_radiation_patter(1, -1, formula=lambda t, p: np.sin(t) ** 4 * np.sin(p) ** 2, plot=plot)
    evaluate_radiation_patter(1, 1, formula=lambda t, p: np.sin(t) ** 4 * np.cos(p) ** 2, plot=plot)
    evaluate_radiation_patter(1, 0, formula=lambda t, p: np.sin(2 * t) ** 2, plot=plot)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("TkAgg")
    test(plot=True)
    plt.show()
