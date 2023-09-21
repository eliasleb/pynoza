import pynoza
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import itertools
from test_EPFL_logo import c_j, plot_current_density
import sympy
import time


def main(**args):
    gamma_si = 4e-9
    gamma = 1
    c0 = 299792458. * gamma_si

    order = args.get("order", 1)
    pause_plot = args.get("pause_plot")
    plot_max = args.get("plot_max")
    # do_singularity_trick = args.get("do_singularity_trick")
    # singularity_region_radius = args["singularity_region_radius"]
    singularity_order = args["singularity_order"]
    # if (singularity_region_radius is None or singularity_order is None) and do_singularity_trick:
    #     raise ValueError(
    #         ":singularity_region_radius: and :singularity_order: are required with option "
    #         ":do_singularity_trick: "
    #     )
    a = 6

    length = 2 * a
    length_logo = 171

    x1s = [0,  0,  10,  0,  0, ]  # 45, 45, 45, 45, 91, 91, 101, 91, 136, 136]
    x2s = [0,  10, 20, 30, 40, ]  # 0,  20, 30, 40,  0, 30, 20,  40,  10, 0]
    ws =  [35, 10, 23, 10, 35, ]  # 10, 23, 10, 23, 10, 10, 23,  35,  10, 35]
    hs =  [10, 10, 10, 10, 10, ]  # 20, 10, 10, 10, 20, 10, 10,  10,  40, 10]
    signs = [1, -4, -1, 4, 1, ]
    polarities = [0, 1, 0, 1, 0, ]
    d1 = a / 4
    d2 = a * 50 / length_logo

    # w0 = 1
    # h0 = 1
    # x10 = np.arange(0, 171, w0)
    # x20 = np.arange(0, 171, h0)
    # for xi, yi in itertools.product(x10, x20):
    #     if 15 ** 2 >= (xi - 68) ** 2 + (yi - 35) ** 2 >= 5 ** 2 and xi >= 68:
    #         x1s.append(xi - w0 / 2)
    #         x2s.append(yi - h0 / 2)
    #         ws.append(w0)
    #         hs.append(h0)

    def current_moment(ax_, ay, az):
        moment = [0, 0, 0]
        if az == 0:
            for x_i, y_i, wi, hi, sign, polarity in zip(x1s, x2s, ws, hs, signs, polarities):
                moment[polarity] += c_j(
                    ax_, ay, az,
                    x_i / length_logo * length - d1,
                    y_i / length_logo * length - d2,
                    wi / length_logo * length,
                    hi / length_logo * length
                ) * sign
        return moment

    if not pause_plot:
        plt.figure()
        plot_current_density(
            x1s,
            x2s,
            ws,
            hs,
            length_logo,
            length,
            d1,
            d2,
        )
        for ind, (x_i, y_i, sign, polarity) in enumerate(zip(x1s, x2s, signs, polarities)):
            plt.text(
                x_i / length_logo * length - d1,
                y_i / length_logo * length - d2,
                f"{ind + 1}: {sign:+d} {['x', 'y', ][polarity]}"
            )

        plt.xlim(-1.1 * a, 1.1 * a)
        plt.ylim(-1.1 * a, 1.1 * a)
        plt.show(block=False)
        time.sleep(1e-6)

    n_pts = 128
    x1 = np.linspace(-.5 * a, .5 * a, n_pts)
    x2 = np.linspace(-.5 * a, .5 * a, n_pts)
    x3 = np.array((0.,))
    t = np.array((0, ))  # np.linspace(-2, 2, 50)  # np.linspace(np.max(x1) - a - 2, np.max(x1) + 5, 1)
    t_sym = sympy.Symbol("t", real=True)
    h_sym = sympy.exp(-(t_sym/sympy.S(gamma))**2)
    # if do_singularity_trick:
    #     d_h_sym = sympy.diff(
    #         h_sym,
    #         t_sym, singularity_order
    #     )
    # else:
    #     d_h_sym = None
#
    sol_causal = pynoza.solution.Solution(
        max_order=order, wave_speed=c0
    )
    sol_causal.recurse()
    sol_causal.set_moments(current_moment=current_moment)
    e_field = sol_causal.compute_e_field(x1, x2, x3, t, h_sym, t_sym)
    # if do_singularity_trick:
    #     e_field_sing = sol_causal.compute_e_field(x1, x2, x3, t, d_h_sym, t_sym)
    # else:
    #     e_field_sing = None
    sol_ac = pynoza.solution.Solution(
        max_order=order, wave_speed=c0, causal=False
    )
    sol_ac.recurse()
    sol_ac.set_moments(current_moment=current_moment)
    e_field_ac = sol_ac.compute_e_field(x1, x2, x3, t, h_sym, t_sym)

    # if do_singularity_trick:
    #     e_field_ac_sing = sol_ac.compute_e_field(x1, x2, x3, t, d_h_sym, t_sym)
    # else:
    #     e_field_ac_sing = None
    plot_data = np.sum((e_field + e_field_ac) ** 2, axis=0) ** .5
    # r = np.sqrt(x1[:, np.newaxis]**2 + x2[np.newaxis, :]**2)
    # if do_singularity_trick:
    #     plot_data_sing = np.sum((
    #                                 (e_field_sing - e_field_ac_sing) * (r/c0)**singularity_order
    #                             ) ** 2, axis=0) ** .5
    #     for i, x_i in enumerate(x1):
    #         for j, y_j in enumerate(x2):
    #             if (x_i**2 + y_j**2)**.5 < singularity_region_radius:
    #                 plot_data[i, j, :] = 0
    #             else:
    #                 plot_data_sing[i, j, :] = 0
    #     plot_data = plot_data + plot_data_sing
    max_data = np.max(plot_data)
    plt.figure(
        figsize=(7, 6)
    )
    plt.ion()
    did_colorbar = False
    for index_t, t_i in enumerate(t):
        plt.gca().clear()
        # max_data = 1e4
        # plt.plot(e_field[2, :, 0, 0, index_t], "k-")
        # plt.plot(e_field_tr[2, :, 0, 0, index_t], "r--")
        # plt.plot(e_field[2, :, 0, 0, index_t] - e_field_tr[2, :, 0, 0, index_t], "g:")
        # plt.ylim(-max_data, max_data)
        if plot_max > 0:
            max_data = plot_max
        # plt.plot(np.abs(e_field[0, :, 0, 0, index_t] - e_field_ac[0, :, 0, 0, index_t]))
        # plt.plot(e_field[0, :, 0, 0, index_t] + e_field[0, :, 0, 0, t.size - 1 - index_t])
        # plt.ylim(-max_data, max_data)
        plt.contourf(
            x1, x2,
            plot_data[:, :, 0, index_t].T,
            levels=np.linspace(0, max_data, 26),
            cmap="hsv"
        )
        if not did_colorbar:
            plt.colorbar()
            did_colorbar = True
        plt.title(f"{t_i=}")
        plt.tight_layout()
        if not pause_plot:
            plt.waitforbuttonpress()
        else:
            plt.show(block=False)
            time.sleep(1)
        plt.savefig(f"data/logo_order-{order}.pdf")


if __name__ == "__main__":
    matplotlib.use("TkAgg")
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "Times"
    })
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", metavar="order", type=int, required=True)
    parser.add_argument('--pause_plot', action=argparse.BooleanOptionalAction)
    parser.add_argument('--do_singularity_trick', action=argparse.BooleanOptionalAction)
    parser.add_argument("--plot_max", type=float, required=False, default=-1)
    parser.add_argument("--singularity_region_radius", type=float)
    parser.add_argument("--singularity_order", type=int)
    parsed = parser.parse_args()
    main(**vars(parsed))

