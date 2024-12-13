import numpy as np
import matplotlib.pyplot as plt
from pynoza import Solution
from itertools import product
from pynoza.from_mathematica import SPHERICAL_TO_CARTESIAN
from pynoza.helpers import get_poynting
from pynoza.solution import fact
from scipy.special import sph_harm


def get_moment(*args, lm=(0, 0), pol=2, coefficients):
    l, m = lm
    if isinstance(l, int):
        d = SPHERICAL_TO_CARTESIAN[lm]
        coefficient = d.get(args, None)
        if coefficient is not None:
            assert np.isreal(coefficient)
            result = [0., 0., 0.]
            result[pol] = coefficient
            return result
        return [0., 0., 0.]
    elif isinstance(l, list):
        result = [0., 0., 0.]
        for li, mi, ci, pol_i in zip(l, m, coefficients, pol):
            d = SPHERICAL_TO_CARTESIAN[(li, mi)]
            coefficient = d.get(args, None)
            if coefficient is not None:
                assert np.isreal(coefficient)
                result[pol_i] += coefficient * ci * (-1) ** np.sum(args) * fact(args)
        return result
    raise ValueError(f"Unknown type for l: {type(l)}")


def evaluate_radiation_pattern(l: int | list, m: int | list, pol: int | list = 2, formula=None, plot=False,
                               coefficients=None, lm_proj=False, do_assert=True):
    c0_si = 3e8
    f0 = 1e9 / c0_si
    t = np.linspace(0, 150e-9, 3_000) * c0_si
    t0 = 70e-9 * c0_si
    sigma = 20e-9 * c0_si
    h = np.sin(2 * np.pi * f0 * t) * np.exp(-1 / 2 * ((t - t0) / sigma)**2)

    if isinstance(l, list):
        print("-------")
        for li, mi, pi in zip(l, m, pol):
            print((li, mi, pi))

    # plt.figure()
    # plt.plot(t, h)
    max_order = l + 2 if isinstance(l, int) else l[0] + 2
    sol = Solution(max_order=max_order)
    sol.recurse()
    sol.set_moments(current_moment=lambda *args: get_moment(*args, lm=(l, m), pol=pol, coefficients=coefficients))

    r = 10
    d_angle = 90 / 10 * np.pi / 180
    theta, phi = np.arange(1e-2, np.pi, d_angle), np.arange(0, 2 * np.pi, d_angle)
    x, y, z = [], [], []
    for theta_i, phi_i in product(theta, phi):
        x.append(r * np.cos(phi_i) * np.sin(theta_i))
        y.append(r * np.sin(phi_i) * np.sin(theta_i))
        z.append(r * np.cos(theta_i))
    x, y, z = np.array(x), np.array(y), np.array(z)
    e_field = sol.compute_e_field(x, y, z, t, h, None, compute_grid=False)
    # pol_show = pol if isinstance(pol, int) else pol[0]
    # ind_max = np.argmax(np.max(np.abs(e_field[pol_show, ...]), axis=-1))
    # plt.plot(t, e_field[pol_show, ind_max, :])
    # plt.show()

    # e_fd = np.fft.fft(e_field, axis=-1)
    # f = np.linspace(0, 1 / (t[1] - t[0]), t.size)
    # plt.plot(f, np.abs(e_fd[0, 10, :]))
    # plt.show()

    b_field = sol.compute_b_field(x, y, z, t, h, None, compute_grid=False)
    # pol_show = 2
    # ind_max = np.argmax(np.max(np.abs(b_field[pol_show, ...]), axis=-1))
    # plt.plot(t, b_field[pol_show, ind_max, :])
    # plt.show()
    poynting = get_poynting(e_field, b_field / (4 * np.pi * 1e-7))
    directivity = np.mean(poynting**2, axis=(0, -1,)).reshape((theta.size, phi.size))**.5
    # directivity = np.mean(e_field ** 2, axis=(0, -1,)).reshape((theta.size, phi.size)) ** .5

    if plot:
        if formula is not None:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
        else:
            plt.figure(figsize=(3, 3))
        plt.contourf(phi*180/np.pi, theta*180/np.pi, directivity,
                     cmap="jet", levels=np.linspace(0, np.max(directivity), 11))
        plt.title(f"{l}, {m}")

    if formula is not None:
        if plot:
            plt.subplot(1, 2, 2)
        analytic = formula(theta[:, None], phi[None, :])
        if isinstance(analytic, np.ndarray):
            if analytic.shape == (theta.size, 1):
                analytic = np.repeat(analytic, phi.size, 1)
            elif analytic.shape == (1, phi.size):
                analytic = np.repeat(analytic, theta.size, 0)
        else:
            analytic = analytic * np.ones((theta.size, phi.size))

        if plot:
            plt.contourf(phi*180/np.pi, theta*180/np.pi, analytic,
                         cmap="jet", levels=np.linspace(0, np.max(analytic), 11))
    if formula is not None:
        directivity /= np.max(directivity)
        analytic /= np.max(analytic)
        if do_assert:
            assert np.allclose(analytic, directivity, atol=5e-4)

    if lm_proj:
        directivity_approx = 0.
        # directivity = sph_harm(0, 1, phi[None, :], theta[:, None])
        error = 1.
        for l0 in range(l + 3 + 1):
            for m0 in range(-l0, l0 + 1):
                old_error = error
                s = sph_harm(m0, l0, phi[None, :], theta[:, None])
                dot = spherical_integral(
                    theta, directivity * np.conjugate(s)) / spherical_integral(theta, s * np.conjugate(s))
                directivity_approx = directivity_approx + dot * s
                error = spherical_integral(
                    theta, np.abs(directivity_approx - directivity)**2) / spherical_integral(
                    theta, np.abs(directivity)**2
                )
                d_error = old_error - error
                if d_error > 1e-6:
                    print(f"{l0=}, {m0=}, error={error*100:.2f} %, d_error={d_error*100:.2f} %")


def spherical_integral(theta, x):
    if len(theta.shape) == 1:
        theta = theta[:, None]
    return np.sum(x * np.sin(theta))


def test_spherical_to_cartesian(plot=False, do_assert=True):
    evaluate_radiation_pattern([0, ], [0, ], pol=[1, ], plot=plot, coefficients=[1., ],
                               formula=lambda t, p: 1 - np.sin(p) ** 2 * np.sin(t) ** 2, do_assert=do_assert)  # 1 -1
    plt.show()
    evaluate_radiation_pattern([0, ], [0, ], pol=[2, ], plot=plot, coefficients=[1., ],
                               formula=lambda t, p: np.sin(t)**2, do_assert=do_assert)  # 1 0
    evaluate_radiation_pattern([0, ], [0, ], pol=[0, ], plot=plot, coefficients=[1., ],
                               formula=lambda t, p: 1 - np.cos(p) ** 2 * np.sin(t) ** 2, do_assert=do_assert)  # 1 1

    evaluate_radiation_pattern(
        [1, 1], [1, -1], pol=[1, 0], plot=plot, coefficients=[1., -1.],
        formula=lambda t, p: np.sqrt(
            (np.cos(2 * p) ** 2 * np.sin(t) ** 2 + .25 * np.sin(2 * t) ** 2 * np.sin(2 * p) ** 2) ** 2),
        do_assert=do_assert)  # 2 -2
    evaluate_radiation_pattern(
        [1, 1], [-1, 0], pol=[2, 1], plot=plot, coefficients=[1, 1], formula=lambda theta, phi:
        np.sqrt(np.maximum(
            1.42483 * np.cos(theta) ** 8 * np.sin(phi) ** 4 +
            (1.42483 * np.sin(theta) ** 8 + 0.534311 * np.sin(2 * theta) ** 4) * np.sin(phi) ** 4 +
            np.cos(theta) ** 6 * (-5.69932 * np.sin(theta) ** 2 * np.sin(phi) ** 4 + 0.712415 * np.sin(2 * phi) ** 2) +
            np.cos(theta) ** 4 * (1.42483 * np.cos(phi) ** 4 - 1.42483 * np.sin(theta) ** 2 * np.sin(2 * phi) ** 2) +
            np.cos(theta) ** 2 * (
                -5.69932 * np.sin(theta) ** 6 * np.sin(phi) ** 4 + 0.712415 * np.sin(theta) ** 4 * np.sin(
                    2 * phi) ** 2),
            0.)),
        do_assert=do_assert
    )  # 2 -1
    evaluate_radiation_pattern([1, 1, 1], [-1, 0, 1], pol=[1, 2, 0], plot=plot, coefficients=[-1, 2, 1],
                               formula=lambda t, _: np.sin(2 * t) ** 2, do_assert=do_assert)  # 2 0
    evaluate_radiation_pattern(
        [1, 1], [0, 1], pol=[0, 2], plot=plot, coefficients=[1, -1], formula=lambda theta, phi:
        np.sqrt(np.maximum(
            1.42483 * np.cos(theta) ** 8 * np.cos(phi) ** 4 + 1.42483 * np.cos(phi) ** 4 * np.sin(theta) ** 8 + np.cos(
                theta) ** 6 * (
                        -5.69932 * np.cos(phi) ** 4 * np.sin(theta) ** 2 + 0.712415 * np.sin(2 * phi) ** 2) + np.cos(
                theta) ** 4 * (
                        8.54897 * np.cos(phi) ** 4 * np.sin(theta) ** 4 + 1.42483 * np.sin(phi) ** 4 - 1.42483 * np.sin(
                            theta) ** 2 * np.sin(2 * phi) ** 2) + np.cos(theta) ** 2 * (
                        -5.69932 * np.cos(phi) ** 4 * np.sin(theta) ** 6 + 0.712415 * np.sin(theta) ** 4 * np.sin(
                            2 * phi) ** 2), 0)), do_assert=do_assert
    )  # 2 1
    evaluate_radiation_pattern(
        [1, 1], [-1, 1], pol=[1, 0], plot=plot, coefficients=[1., 1.], formula=lambda theta, phi:
        np.sin(theta) ** 2 * np.sqrt(np.maximum(
            1.42483 * np.cos(theta) ** 4 * np.cos(2. * phi) ** 4 + 2.84966 * np.cos(theta) ** 2 * np.cos(
                2. * phi) ** 2 * np.sin(2. * phi) ** 2 + 1.42483 * np.sin(2. * phi) ** 4, 0)), do_assert=do_assert
    )  # 2, 2

    evaluate_radiation_pattern(
        [2, 2], [-2, 2], pol=[0, 1], plot=plot, coefficients=[1., 1.], formula=lambda theta, phi:
        np.sin(theta) ** 4 * np.sqrt(np.maximum(
            9.81796 * np.cos(3. * phi) ** 4 + 19.6359 * np.cos(theta) ** 2 * np.cos(3. * phi) ** 2 * np.sin(
                3. * phi) ** 2 + 9.81796 * np.cos(theta) ** 4 * np.sin(3. * phi) ** 4, 0)), do_assert=do_assert
    )  # 3, -3
    evaluate_radiation_pattern(
        [2, 2, 2], [-2, -1, 1], pol=[2, 0, 1, ],
        plot=plot, coefficients=[1., 1., -1.], formula=lambda theta, phi:
        np.sqrt(np.maximum(
            (1.63633 - 2.18177 * np.cos(4 * theta) + 0.545442 * np.cos(8 * theta)) * np.cos(2. * phi) ** 4 +
            (0.779812 - 1.00566 * np.cos(2 * theta) + 0.884213 * np.cos(4 * theta) - 0.946002 * np.cos(6 * theta) +
             0.345163 * np.cos(8 * theta) - 0.230109 * np.cos(10 * theta) + 0.172581 * np.cos(12 * theta)) * np.sin(
                2. * phi) ** 4 +
            (0.238631 + 0.0170451 / np.sin(2 * theta) * np.sin(4 * theta) - 0.0681803 / np.sin(4 * theta) * np.sin(
                8 * theta) -
             0.0937479 / np.sin(6 * theta) * np.sin(12 * theta) - 0.0511352 / np.sin(8 * theta) * np.sin(16 * theta) +
             0.0767028 / np.sin(10 * theta) * np.sin(20 * theta)) * np.sin(4. * phi) ** 2,
            0)), do_assert=do_assert
    )  # 3, -2
    evaluate_radiation_pattern(
        [2, 2, 2, 2], [-2, -1, 0, 2],
        pol=[0, 2, 1, 1, ], plot=plot,
        coefficients=[-0.167332, 0.669328, 0.579655, 0.167332], formula=lambda theta, phi:
        0.884522 * np.abs(
            1.46863 + np.cos(2 * theta) + 0.479705 * np.cos(4 * theta) + 0.830258 * np.cos(6 * theta) +
            0.385609 * np.cos(2 * theta - 2 * phi) + 0.129151 * np.cos(4 * theta - 2 * phi) -
            0.415129 * np.cos(6 * theta - 2 * phi) - 0.199262 * np.cos(2 * phi) +
            0.385609 * np.cos(2 * (theta + phi)) + 0.129151 * np.cos(4 * theta + 2 * phi) -
            0.415129 * np.cos(6 * theta + 2 * phi)), do_assert=do_assert
    )  # 3, -1
    evaluate_radiation_pattern(
        [2, 2, 2, ], [-1, 0, 1, ],
        pol=[1, 2, 0, ], plot=plot,
        coefficients=[-0.409878, 0.70993, 0.409878, ], formula=lambda theta, phi:
        15 * np.abs(0.2 - np.cos(theta) ** 2) * np.sin(theta) * np.sqrt(np.maximum(
            (0.174542 + 4.36354 * np.cos(theta) ** 4) * np.sin(theta) ** 2 - 0.436354 * np.sin(2 * theta) ** 2, 0)),
        do_assert=do_assert
    )  # 3, 0
    evaluate_radiation_pattern(
        [2, 2, 2, 2, ], [-2, 0, 1, 2],
        pol=[1, 0, 2, 0], plot=plot,
        coefficients=[0.167332, -0.579655, 0.669328, 0.167332], formula=lambda theta, phi:
        0.884522 * np.abs(
            1.46863 + np.cos(2 * theta) + 0.479705 * np.cos(4 * theta) + 0.830258 * np.cos(6 * theta) -
            0.385609 * np.cos(2 * theta - 2 * phi) - 0.129151 * np.cos(4 * theta - 2 * phi) +
            0.415129 * np.cos(6 * theta - 2 * phi) + 0.199262 * np.cos(2 * phi) -
            0.385609 * np.cos(2 * (theta + phi)) - 0.129151 * np.cos(4 * theta + 2 * phi) +
            0.415129 * np.cos(6 * theta + 2 * phi)), do_assert=do_assert
    )  # 3, 1
    evaluate_radiation_pattern(
        [2, 2, 2, ], [-1, 1, 2, ],
        pol=[1, 0, 2, ], plot=plot,
        coefficients=[-0.52915, -0.52915, 0.52915], formula=lambda theta, phi:
        8.35563 * np.sin(theta) ** 2 * (
                np.cos(theta) ** 4 * np.cos(2. * phi) ** 2 +
                0.25 * np.cos(2. * phi) ** 2 * np.sin(theta) ** 4 +
                np.cos(theta) ** 2 * (-np.cos(2. * phi) ** 2 * np.sin(theta) ** 2 + np.sin(2. * phi) ** 2)
        ), do_assert=do_assert
    )  # 3, 2
    evaluate_radiation_pattern(
        [2, 2, ], [-2, 2, ],
        pol=[1, 0, ], plot=plot,
        coefficients=[0.648074, -0.648074, ], formula=lambda theta, phi:
        np.sin(theta) ** 4 * np.sqrt(np.maximum(
            9.81796 * np.cos(theta) ** 4 * np.cos(3. * phi) ** 4 +
            19.6359 * np.cos(theta) ** 2 * np.cos(3. * phi) ** 2 * np.sin(3. * phi) ** 2 +
            9.81796 * np.sin(3. * phi) ** 4, 0)), do_assert=do_assert
    )  # 3, 3

    evaluate_radiation_pattern(
        [3, 3, ], [-3, 3, ],
        pol=[1, 0, ], plot=plot,
        coefficients=[-10.6066, -10.6066, ], formula=lambda theta, phi:
        np.sin(theta) ** 6 * np.sqrt(np.maximum(
            39.2719 * np.cos(theta) ** 4 * np.cos(4. * phi) ** 4 +
            78.5437 * np.cos(theta) ** 2 * np.cos(4. * phi) ** 2 * np.sin(4. * phi) ** 2 +
            39.2719 * np.sin(4. * phi) ** 4, 0)), do_assert=do_assert
    )  # 4, 4


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("TkAgg")
    test_spherical_to_cartesian(plot=True, do_assert=True)
    # test_cartesian_to_spherical(plot=True)
    plt.show()
