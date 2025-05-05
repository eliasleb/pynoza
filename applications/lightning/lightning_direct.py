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


def channel_base_current(t, for_plot=False):
    ind_nonzero = t > 0
    tn0 = t[ind_nonzero]

    heidler = np.zeros(t.shape)

    if for_plot:
        peak_current = 1
    else:
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
                   direction=1, plot_name=""):
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
            # if i % 100 == 0:
            #     print(f"{i/t.size * 100:.1f}")
        # hi = gaussian_filter(hi, 100, axes=-1)
        h_dict[(0, 0, az)] = [
            0. * hi, 0. * hi, hi
        ]
        plt.plot(t, hi)

    return h_dict


def get_h(t, max_order=0, unit_time=0., unit_distance=0., channel_height_m=0., attenuation=lambda *args: 0., z0=0.,
          v=0.5, symmetry=True, plot_case=""):
    v = v * 3e8 / unit_distance * unit_time
    dt = t[1] - t[0]
    dz = dt * v
    z_physical = np.arange(-100 / unit_distance, (channel_height_m + 100) / unit_distance, dz)
    if z0 >= 0.:
        z_moment = z_physical - z0
    else:
        z_moment = -z_physical - z0

    current = attenuation(z_physical)[None, :] * channel_base_current(
        (t[:, None] - z_physical[None, :] / v) * unit_time
    )

    plt.figure(figsize=(4, 3))
    plt.contourf(t * unit_time * 1e6, z_moment * unit_distance, current.T / 1e3, cmap="jet", levels=21)
    plt.xlabel("Time (μs)")
    plt.ylabel("Altitude z (m)")
    plt.title("Channel base current (kA)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{plot_case}_channel_base_current.pdf")

    plt.figure(figsize=(4, 3))
    h_dict = dict()
    step = 2 if symmetry else 1
    cmap = plt.get_cmap("jet")
    for ind, az in enumerate(range(0, max_order + 1, step)):
        hi = dz * np.trapezoid(
            z_moment ** az * current,
            axis=1
        )
        h_dict[(0, 0, az)] = [
            0. * hi, 0. * hi, hi
        ]
        plt.loglog(t * unit_time * 1e6, hi * unit_distance ** (az + 1),
                     color=cmap(ind/(max_order + 1)), label=f"a_z = {az}")

    plt.legend(loc="lower right")
    plt.xlim(np.min(t[t > 0] * unit_time * 1e6), np.max(t * unit_time * 1e6))
    plt.xlabel("Time (μs)")
    plt.ylabel("")
    plt.title("Time-dependent moments")
    plt.tight_layout()
    plt.savefig(f"{plot_case}_moments.pdf")
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


def read_matlab(r):
    res = loadmat(f"../../../lightning_inverse/Campi_Analitici_v2024/lightpesto_r{int(r)}.mat")
    return res["t"].squeeze(), res["h_phi"].squeeze(), res["e_z"].squeeze(), res["e_r"].squeeze()


def signed_log(y, log10_threshold=-12):
    too_small = np.abs(y) < 10**log10_threshold
    ok = np.invert(too_small)
    y[too_small] = 0
    y[ok] = np.sign(y[ok]) * np.log10(np.abs(y[ok]) / 10**log10_threshold)
    # y[ok][y[ok] > 0] += np.log10(threshold)
    return y


def plot_heidler_derivatives(max_order=3):
    log10_threshold = -12
    t = np.linspace(-2, 10, 2000) * 1e-6
    dt = t[1] - t[0]
    heidler_s = [channel_base_current(t, for_plot=True), ]
    for order in range(1, max_order + 1):
        heidler_s.append(np.gradient(heidler_s[-1]) / dt)

    _, ax = plt.subplots(figsize=(5, 3))
    linestyles = ["-", "--", ":", "-.", "-o"]
    cmap = plt.get_cmap("jet")
    for ind, (order, linestyle) in enumerate(zip(range(max_order + 1), linestyles)):
        plt.plot(t * 1e6, signed_log(heidler_s[ind], log10_threshold=log10_threshold), linestyle, label=f"n = {order}",
                 color=cmap(ind/(max_order + 1))
                 )

    tick_values = range(-30, 30 + 1, 10)
    tick_labels = [
        rf"$-10^{{{-e + log10_threshold}}}$" if e < 0 else rf"$10^{{{e + log10_threshold}}}$"
        for e in tick_values
    ]
    ax.set_yticks(tick_values)
    ax.set_yticklabels(tick_labels, fontname='Times New Roman')
    plt.xlim(-2, 9.9)
    plt.xlabel("Time (us)")
    plt.ylabel("A/s^n")
    plt.title("Channel base current derivatives")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../../../lightning_inverse/figs/heidler_derivatives_v2.pdf")


def lightning_multipole_expansion(r_m=8000, max_order=2, block_plot=False, t_max_us=None):

    plot_case = f"../../../lightning_inverse/figs/field_r_{int(r_m)}_order_{max_order}"

    plot_heidler_derivatives()

    # attenuation_sym = lambda z_: sympy.exp(-abs(z_) / 2_000 * unit_distance)
    channel_height_m = 4000
    unit_distance = channel_height_m
    unit_time = 1 / 3e8 * unit_distance
    # unit_time_sym = 1 / sympy.S(3) / sympy.S(10) ** 8 * unit_distance

    c0 = 3e8
    c = c0 / unit_distance * unit_time
    print(f"{c=}")

    r = r_m / unit_distance
    z = 10 / unit_distance
    z0 = 0 / unit_distance
    v = .5

    t = np.arange(0, 80, .1) * 1e-6 / unit_time
    channel_delay = channel_height_m / c0 / v
    t_sing1 = channel_delay + r * unit_distance / c0
    t_sing2 = np.sqrt(channel_height_m ** 2 + (r * unit_distance) ** 2) / c0 + channel_delay
    t_early = min(r * unit_distance / c0 / v, channel_height_m / c0 / v)
    print(f"{t.shape=}")
    print(f"Channel delay: {channel_delay * 1e6:.2f} us => {t_sing1 * 1e6:.2f} us, {t_sing2 * 1e6:.2f} us")
    print(f"t early = {t_early * 1e6:.2f} us")
    x1 = np.array([r, ])
    x2 = np.array([0, ])
    x3_physical = np.array([z, ])

    current = channel_base_current(t * unit_time)
    t_max = t[np.argmax(current)]
    print(f"t max: {t_max * unit_time * 1e6:.2f} us")
    print(f"r min: {t_max * v * unit_distance:.2f} m")

    attenuation = lambda z_: np.exp(-np.abs(z_) / 2000 * unit_distance) \
                             * (z_ >= 0.) * (z_ * unit_distance <= channel_height_m)

    has_image = np.abs(z0) > 1e-12

    sol = Solution(max_order=max_order + 2, wave_speed=c)
    sol.recurse()
    h = get_h(t, max_order=max_order, unit_time=unit_time, unit_distance=unit_distance,
              channel_height_m=channel_height_m, attenuation=attenuation, z0=z0, v=v, symmetry=not has_image,
              plot_case=plot_case)
    if has_image:
        h_img = get_h(t, max_order=max_order, unit_time=unit_time, unit_distance=unit_distance,
                      channel_height_m=channel_height_m, attenuation=attenuation, z0=-z0, v=v, symmetry=not has_image)
    args = (x1, x2, x3_physical - z0, t, h)
    if has_image:
        args_img = (x1, x2, x3_physical + z0, t, h_img)
    kwargs = dict(
            delayed=True
    )
    start_time = time.perf_counter()
    e_field = sol.compute_e_field(*args, **kwargs) / unit_time
    b_field = sol.compute_b_field(*args, **kwargs) / unit_distance
    if has_image:
        e_field_img = sol.compute_e_field(*args_img, **kwargs) / unit_time
        b_field_img = sol.compute_b_field(*args_img, **kwargs) / unit_distance
    end_time = time.perf_counter()

    # print(f"{np.max(np.abs(b_field[1, ...]))*1e6:.2f}")

    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time:.4f} seconds")
    t_comp, h_phi, e_z, e_r = read_matlab(r * unit_distance)
    t_comp *= 1e6
    ts_us = t_early * 1e6, t_sing1 * 1e6, t_sing2 * 1e6

    plt.figure(figsize=(6, 6))
    linestyle_ref = "k-"
    linestyle_me = "r--"
    plt.subplot(3, 1, 1)
    plt.plot(t_comp, h_phi * 1e3, linestyle_ref, label="Ref.")
    plt.plot(t * unit_time * 1e6, -2 * b_field[1, 0, 0, 0, :] / (4 * np.pi * 1e-7), linestyle_me, label="M.E.")
    if has_image:
        plt.plot(t * unit_time * 1e6, -2 * b_field_img[1, 0, 0, 0, :] / (4 * np.pi * 1e-7), label="2")
        plt.plot(t * unit_time * 1e6, -(b_field + b_field_img)[1, 0, 0, 0, :] / (4 * np.pi * 1e-7), label="both")
    plt.title("H, azimuthal (A/m)")
    y1, y2 = -np.max(np.abs(h_phi)) * 1e3 / 10, np.max(np.abs(h_phi)) * 1e3 * 1.1
    plt.ylim(y1, y2)
    plt.vlines(ts_us, y1, y2, colors="b", linestyles=":")
    plt.xlim(np.min(t) * unit_time * 1e6, np.max(t) * unit_time * 1e6 if t_max_us is None else t_max_us)
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(t_comp, e_z * 1e3, linestyle_ref, label="ref")
    plt.plot(t * unit_time * 1e6, 2 * e_field[2, 0, 0, 0, :], linestyle_me, label="ME")
    if has_image:
        plt.plot(t * unit_time * 1e6, 2 * e_field_img[2, 0, 0, 0, :], label="2")
        plt.plot(t * unit_time * 1e6, (e_field + e_field_img)[2, 0, 0, 0, :], label="both")
    plt.title("E, vertical (V/m)")
    y1, y2 = -np.max(np.abs(e_z)) * 1e3 * 1.1, np.max(np.abs(e_z)) * 1e3 * .1
    plt.ylim(y1, y2)
    plt.vlines(ts_us, y1, y2, colors="b", linestyles=":")
    plt.xlim(np.min(t) * unit_time * 1e6, np.max(t) * unit_time * 1e6 if t_max_us is None else t_max_us)
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(t_comp, e_r * 1e3, linestyle_ref, label="Ref.")
    plt.plot(t * unit_time * 1e6, 2 * e_field[0, 0, 0, 0, :], linestyle_me, label="M.E.")
    if has_image:
        plt.plot(t * unit_time * 1e6, e_field_img[0, 0, 0, 0, :], label="2")
        plt.plot(t * unit_time * 1e6, (e_field + e_field_img)[0, 0, 0, 0, :], label="both")
    plt.title("E, radial (V/m)")
    y1, y2 = -np.max(np.abs(e_r)) * 1e3 / 10, np.max(np.abs(e_r)) * 1e3 * 1.1
    plt.ylim(y1, y2)
    plt.legend()
    plt.xlabel("Time (us)")
    plt.xlim(np.min(t) * unit_time * 1e6, np.max(t) * unit_time * 1e6 if t_max_us is None else t_max_us)
    plt.vlines(ts_us, y1, y2, colors="b", linestyles=":")
    plt.grid()

    plt.tight_layout()
    plt.savefig(f"{plot_case}_fields.pdf")

    max_h_phi_ref = np.max(1e3 * h_phi)
    max_h_phi_com = np.max(-2 * b_field[1, 0, 0, 0, t < t_early / unit_time] / (4 * np.pi * 1e-7))
    error = (max_h_phi_com - max_h_phi_ref) / max_h_phi_ref
    print(f"Relative peak error: {error * 100:.2f} %")

    if block_plot:
        plt.show()
    print("------------------------------------------")


def main():
    lightning_multipole_expansion(r_m=8000, max_order=10, block_plot=True)
    lightning_multipole_expansion(r_m=3000, max_order=10, block_plot=True, t_max_us=30)
    for max_order in range(0, 10 + 1, 2):
        lightning_multipole_expansion(r_m=3000, max_order=max_order, block_plot=False, t_max_us=30)
        plt.close("all")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("TkAgg")
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    main()
