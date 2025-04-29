import numpy as np
from pynoza import Solution
import matplotlib.pyplot as plt
import time
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter, convolve1d
import multiprocessing
from scipy.special import erfc
from scipy.signal import savgol_filter, sosfilt, butter
import sympy
import sympy.functions
from scipy.integrate import quad


def channel_base_current(t):
    ind_nonzero = t > 0
    tn0 = t[ind_nonzero]

    heidler = np.zeros(t.shape)

    peak_current = 30e3
    i01 = 0.940493940709144
    tau11 = 1.8e-6
    tau12 = 95.0e-6
    n1 = 2.0e0
    argo1 = (tau11 / tau12) * (n1 * tau12 / tau11) ** (1. / n1)
    eta1 = np.exp(-argo1)
    x1 = (tn0 / tau11) ** n1
    term1 = x1 / (1 + x1)
    heidler[ind_nonzero] = (i01 * term1 / eta1 * np.exp(-tn0 / tau12))
    heidler = peak_current * heidler
    # plt.plot(t, heidler)
    return heidler


def symbolic_channel_base_current(t):
    tau1 = sympy.S(18)/sympy.S(10)
    tau2 = sympy.S(95)
    n = 2
    argo1 = (tau1 / tau2) * (n * tau2 / tau1) ** (1 / n)
    eta1 = sympy.exp(-argo1)
    x1 = (t / tau1) ** n
    term1 = x1 / (1 + x1)
    return 30 * term1 / eta1 * sympy.exp(-t / tau2) * sympy.Heaviside(t)


def get_h_symbolic(t, max_order=0, unit_time=0., unit_distance=0., channel_height_m=0., attenuation=lambda *args: 0,
                   direction=1):
    v = sympy.S(1) / 2
    z = sympy.S("z")
    t_sym = sympy.S("t")
    current = symbolic_channel_base_current((t_sym - direction * (z + direction * sympy.S(1) / 2) / v)
                                            * unit_time * sympy.S(10) ** 6) \
              * attenuation(z + sympy.S(1) / 2)
    h_dict = dict()
    plt.figure()
    for az in range(0, max_order + 1):
        hi = np.zeros(t.shape)
        for i, ti in enumerate(t):
            integrand = sympy.lambdify(z, (current * z ** az).subs(t_sym, ti), modules=["numpy"])
            res, _error = quad(integrand, -.5, .5)
            hi[i] = res * 1e3
            if i % 100 == 0:
                print(f"{i/t.size * 100:.1f}")
        # hi = gaussian_filter(hi, 100, axes=-1)
        h_dict[(0, 0, az)] = [
            0. * hi, 0. * hi, hi
        ]
        plt.plot(t, hi)
    return h_dict


def get_h(t, max_order=0, unit_time=0., unit_distance=0., channel_height_m=0., attenuation=lambda *args: 0., z0=0.):
    v = 0.5 * 3e8 / unit_distance * unit_time
    dt = t[1] - t[0]
    dz = dt * v
    z_physical = np.arange(-100 / unit_distance, (channel_height_m + 100) / unit_distance, dz)
    if z0 >= 0.:
        z_moment = z_physical - z0
    else:
        z_moment = -z_physical - z0

    plt.figure()
    plt.plot(z_physical, z_moment)
    plt.xlabel("phys")
    plt.ylabel("virt")

    current = attenuation(z_physical)[None, :] * channel_base_current(
        (t[:, None] - z_physical[None, :] / v) * unit_time
    )
    plt.figure()
    plt.contourf(t * unit_time * 1e6, z_moment, current.T, cmap="jet", levels=21)
    plt.colorbar()

    plt.figure()
    h_dict = dict()
    for az in range(0, max_order + 1, 1):
        hi = dz * np.trapezoid(
            z_moment ** az * current,
            axis=1
        )
        h_dict[(0, 0, az)] = [
            0. * hi, 0. * hi, hi
        ]
        plt.plot(t * unit_time * 1e6, hi, label=f"a_z = {az}")

    plt.legend()
    plt.xlabel("Time (us)")
    plt.ylabel("Moment (a.u.)")
    return h_dict


def get_scalar_moments(ax, ay, az, unit_distance=0., channel_height_m=0., attenuation=lambda *args: 0.):
    z = np.linspace(0, channel_height_m / unit_distance, 100)
    z_moment = z - channel_height_m / unit_distance / 2.
    dz = z_moment[1] - z_moment[0]
    if ax == 0 and ay == 0 and az % 2 == 0:
        return [0., 0.,
                2 * dz * np.trapezoid(z_moment ** az * attenuation(z))
                ]
    return [0., 0., 0.]


def read_matlab():
    res = loadmat("../../../lightning_inverse/Campi_Analitici_v2024/lightpesto.mat")
    return res["t"].squeeze(), res["h_phi"].squeeze(), res["e_z"].squeeze(), res["e_r"].squeeze()


def main():
    max_order = 2

    # attenuation_sym = lambda z_: sympy.exp(-abs(z_) / 2_000 * unit_distance)
    channel_height_m = 4000
    unit_distance = channel_height_m
    unit_time = 1 / 3e8 * unit_distance
    # unit_time_sym = 1 / sympy.S(3) / sympy.S(10) ** 8 * unit_distance

    c0 = 3e8
    c = c0 / unit_distance * unit_time
    print(f"{c=}")
    r = 3000 / unit_distance
    z = 10 / unit_distance
    z0 = 1000 / unit_distance
    t = np.arange(0, 30, .1) * 1e-6 / unit_time
    print(f"{t.shape=}")
    x1 = np.array([r, ])
    x2 = np.array([0, ])
    x3_physical = np.array([z, ])

    attenuation = lambda z_: np.exp(-np.abs(z_) / 2000 * unit_distance) \
                             * (z_ >= 0.) * (z_ <= channel_height_m)

    sol = Solution(max_order=max_order + 2, wave_speed=c)
    sol.recurse()
    h = get_h(t, max_order=max_order, unit_time=unit_time, unit_distance=unit_distance,
              channel_height_m=channel_height_m, attenuation=attenuation, z0=z0)
    h_img = get_h(t, max_order=max_order, unit_time=unit_time, unit_distance=unit_distance,
                  channel_height_m=channel_height_m, attenuation=attenuation, z0=-z0)
    args = (x1, x2, x3_physical - z0, t, h)
    args_img = (x1, x2, x3_physical + z0, t, h_img)
    kwargs = dict(
            delayed=True
    )
    start_time = time.perf_counter()
    e_field = sol.compute_e_field(*args, **kwargs) / unit_time
    b_field = sol.compute_b_field(*args, **kwargs) / unit_distance
    e_field_img = sol.compute_e_field(*args_img, **kwargs) / unit_time
    b_field_img = sol.compute_b_field(*args_img, **kwargs) / unit_distance
    end_time = time.perf_counter()

    print(f"{np.max(np.abs(b_field[1, ...]))*1e6:.2f}")

    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time:.4f} seconds")
    t_comp, h_phi, e_z, e_r = read_matlab()
    t_comp *= 1e6

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(t_comp, h_phi * 1e3, "--", label="comp.")
    plt.plot(t * unit_time * 1e6, -b_field[1, 0, 0, 0, :] / (4 * np.pi * 1e-7), label="1")
    plt.plot(t * unit_time * 1e6, -b_field_img[1, 0, 0, 0, :] / (4 * np.pi * 1e-7), label="2")
    plt.plot(t * unit_time * 1e6, -(b_field + b_field_img)[1, 0, 0, 0, :] / (4 * np.pi * 1e-7), label="both")
    plt.title("H, azimuthal (A/m)")
    plt.ylim(-1e-1, 1.1 * np.max(h_phi) * 1e3)

    plt.subplot(3, 1, 2)
    plt.plot(t_comp, e_z * 1e3, "--", label="comp.")
    plt.plot(t * unit_time * 1e6, e_field[2, 0, 0, 0, :], label="1")
    plt.plot(t * unit_time * 1e6, e_field_img[2, 0, 0, 0, :], label="2")
    plt.plot(t * unit_time * 1e6, (e_field + e_field_img)[2, 0, 0, 0, :], label="both")
    plt.title("E, vertical (V/m)")
    plt.ylim(1.1 * np.min(e_z) * 1e3, 50)

    plt.subplot(3, 1, 3)
    plt.plot(t_comp, e_r * 1e3, "--", label="comp.")
    plt.plot(t * unit_time * 1e6, e_field[0, 0, 0, 0, :], label="1")
    plt.plot(t * unit_time * 1e6, e_field_img[0, 0, 0, 0, :], label="2")
    plt.plot(t * unit_time * 1e6, (e_field + e_field_img)[0, 0, 0, 0, :], label="both")
    plt.title("E, radial (V/m)")
    plt.ylim(-.1, 1.1 * np.max(e_r) * 1e3)
    plt.legend()
    plt.xlabel("Time (us)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("TkAgg")
    main()
