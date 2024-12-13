import numpy as np
from pynoza import Solution
import matplotlib.pyplot as plt
import time
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
import multiprocessing
from scipy.special import erfc
from scipy.signal import savgol_filter, sosfilt, butter


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
    return peak_current * heidler


def get_h(t, max_order=0, unit_time=0., unit_distance=0., channel_height_m=0., attenuation=lambda *args: 0., r=0.):
    v = 0.5 * 3e8 / unit_distance * unit_time
    dt = t[1] - t[0]
    # bound = np.clip(
    #     (t - r / v) * v,
    #     0,
    #     channel_height_m / unit_distance,
    # )
    dz = dt * v
    z_moment = np.arange(0, channel_height_m / unit_distance, dz)
    # smoothing = .5 * erfc(40 * (z_moment / np.max(z_moment) - .9))

    # z_moment_2d = z_moment[None, :] * (z_moment[None, :] <= bound[:, None])
    # plt.contourf(z_moment_2d.T, cmap="jet")
    current = (attenuation(z_moment) * 1)[None, :] * channel_base_current(
        (t[:, None] - z_moment[None, :] / v) * unit_time
    )
    # current = gaussian_filter(current, int(.1 / dt))
    h_dict = dict()
    # plt.plot(t, channel_base_current(t * unit_time))
    # ind_smooth = t * unit_time > 16e-6
    # lowpass = butter(
    #     2,
    #     .1,
    #     output="sos"
    # )
    for az in range(0, max_order + 1, 2):
        hi = 2 * dz * np.trapezoid(
            z_moment ** az * current,
            axis=1
        )
        # hi = sosfilt(lowpass, hi)  # savgol_filter(hi, 10, 9)
        h_dict[(0, 0, az)] = [
            0 * hi, 0 * hi, hi
        ]
        # if az == 2:
        #     plt.plot(t, hi)
        #     plt.plot(t[1:], np.diff(hi) / dt)
        #     plt.plot(t[2:], np.diff(hi, 2) / dt**2)
        #     plt.plot(t[3:], np.diff(hi, 3) / dt**3)
        #     plt.plot(t[4:], np.diff(hi, 4) / dt**4)

    return h_dict


def get_scalar_moments(ax, ay, az, unit_distance=0., channel_height_m=0., attenuation=lambda *args: 0.):
    z_moment = np.linspace(0, channel_height_m / unit_distance, 100)
    dz = z_moment[1] - z_moment[0]
    if ax == 0 and ay == 0 and az % 2 == 0:
        return [0., 0.,
                2 * dz * np.trapezoid(z_moment ** az * attenuation(z_moment))
                ]
    return [0., 0., 0.]


def read_matlab():
    res = loadmat("../../../lightning_inverse/lightpesto.mat")
    return res["t"].squeeze(), res["h_phi"].squeeze(), res["e_z"].squeeze(), res["e_r"].squeeze()


def main():
    max_order = 4

    channel_height_m = 4000.
    attenuation = lambda z_: np.exp(-np.abs(z_) / 2_000 * unit_distance)

    unit_distance = 4000.
    unit_time = 1 / 3e8 * unit_distance

    c0 = 3e8
    c = c0 / unit_distance * unit_time
    print(f"{c=}")
    # n_t = 8000
    r = 2e3 / unit_distance
    z = 10 / unit_distance
    # t_max = dt * (n_t - 1)
    t = np.arange(0, 40, 1e-2) * 1e-6 / unit_time
    print(f"{t.shape=}")
    x1 = np.array([r, ])
    x2 = np.array([0, ])
    x3 = np.array([z, ])

    sol = Solution(max_order=max_order + 2, wave_speed=c)
    sol_tr = Solution(max_order=max_order + 2, wave_speed=c, causal=False)
    sol.recurse()
    sol_tr.recurse()
    h = get_h(t, max_order=max_order, unit_time=unit_time, unit_distance=unit_distance,
              channel_height_m=channel_height_m, attenuation=attenuation, r=np.sqrt(z**2 + r**2))
    # h_tr = get_h(t, max_order=max_order, unit_time=unit_time, unit_distance=unit_distance,
    #              channel_height_m=channel_height_m, attenuation=attenuation, r=np.sqrt(z**2 + r**2))
    args = (x1, x2, x3, t, h)
    kwargs = dict(
            delayed=True
    )
    start_time = time.perf_counter()
    # with multiprocessing.Pool(processes=2) as pool:
    #     e_field = pool.apply_async(sol.compute_e_field, args=args, kwds=kwargs)
    #     b_field = pool.apply_async(sol.compute_b_field, args=args, kwds=kwargs)
    #     e_field = e_field.get() / unit_time
    #     b_field = b_field.get() / unit_distance
    e_field = sol.compute_e_field(*args, **kwargs) / unit_time
    b_field = sol.compute_b_field(*args, **kwargs) / unit_distance
    # b_field_tr = sol_tr.compute_b_field(*args, **kwargs) / unit_distance
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time:.4f} seconds")
    t_comp, h_phi, e_z, e_r = read_matlab()
    t_comp *= 1e6

    plt.figure()

    plt.subplot(3, 1, 1)
    plt.plot(t_comp, h_phi * 1e3, "--")
    plt.plot(t * unit_time * 1e6, -b_field[1, 0, 0, 0, :] / (4 * np.pi * 1e-7))
    # plt.plot(t * unit_time * 1e6, -b_field_tr[1, 0, 0, 0, :] / (4 * np.pi * 1e-7))
    # plt.plot(t * unit_time * 1e6, -.5 * (b_field - b_field_tr)[1, 0, 0, 0, :] / (4 * np.pi * 1e-7))
    plt.ylim(-1e-1, 1.1 * np.max(h_phi) * 1e3)

    plt.subplot(3, 1, 2)
    plt.plot(t_comp, e_z * 1e3, "--")
    plt.plot(t * unit_time * 1e6, e_field[2, 0, 0, 0, :])
    plt.ylim(1.1 * np.min(e_z) * 1e3, 50)

    plt.subplot(3, 1, 3)
    plt.plot(t_comp, e_r * 1e3, "--")
    plt.plot(t * unit_time * 1e6, e_field[0, 0, 0, 0, :])
    plt.ylim(-.1, 1.1 * np.max(e_r) * 1e3)

    plt.show()


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("TkAgg")
    main()
