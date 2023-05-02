import pynoza
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import itertools
from test_EPFL_logo import c_j
import sympy
import time


def main(**args):
    gamma_si = 4e-9
    gamma = 1
    c0 = 299792458. * gamma_si

    order = args.get("order", 1)
    a = 5

    length = 2 * a
    length_logo = 171

    x1s = [0, 0, 10, 0, 0, 45, 45, 45, 45, 91, 91, 101, 91, 136, 136]
    x2s = [0, 10, 20, 30, 40, 0, 20, 30, 40, 0, 30, 20, 40, 10, 0]
    ws = [35, 10, 23, 10, 35, 10, 23, 10, 23, 10, 10, 23, 35, 10, 35]
    hs = [10, 10, 10, 10, 10, 20, 10, 10, 10, 20, 10, 10, 10, 40, 10]

    d1 = a
    d2 = a * 50 / length_logo

    w0 = 1
    h0 = 1
    x10 = np.arange(0, 171, w0)
    x20 = np.arange(0, 171, h0)
    for xi, yi in itertools.product(x10, x20):
        if 15 ** 2 >= (xi - 68) ** 2 + (yi - 35) ** 2 >= 5 ** 2 and xi >= 68:
            x1s.append(xi - w0 / 2)
            x2s.append(yi - h0 / 2)
            ws.append(w0)
            hs.append(h0)

    t = np.linspace(19, 20, 1)

    def modulation(x, y):
        return np.sign(np.sin(2 * np.pi * x / 50 * 3) * np.sin(2 * np.pi * y / 171 * 10))

    def current_moment(ax_, ay, az):
        moment = 0
        if az == 0:
            for x_i, y_i, wi, hi in zip(x1s, x2s, ws, hs):
                moment += c_j(ax_, ay, az, x_i / length_logo * length - d1,
                              y_i / length_logo * length - d2, wi / length_logo * length,
                              hi / length_logo * length) / gamma_si * modulation(x_i, y_i)
        return [0, 0, moment]

    n_pts = 128
    x1 = np.linspace(-1.5 * a, 1.5 * a, n_pts)
    x2 = np.linspace(3 * a, 6 * a, n_pts)
    x3 = np.array((0.,))
    t_sym = sympy.Symbol("t", real=True)
    # h_sym = (3 * gamma_si * sympy.sqrt(np.pi / 2)) ** -.5 \
    #         * sympy.exp(-((t_sym-3*gamma)/gamma)**2) * (4*((t_sym-3*gamma)/gamma)**2-2)
    h_sym = sympy.exp(-(t_sym/sympy.S(gamma))**2)
    # h_sym = sympy.exp(-(t_sym/gamma)**2)
    # h_sym_tr = h_sym.subs(t_sym, -t_sym)

    sol_causal = pynoza.solution.Solution(max_order=order, wave_speed=c0)
    sol_causal.recurse()
    sol_causal.set_moments(current_moment=current_moment)
    e_field = sol_causal.compute_e_field(x1, x2, x3, t, h_sym, t_sym)
    # sol_anticausal = pynoza.solution.Solution(max_order=order, wave_speed=c0, causal=False)
    # sol_anticausal.recurse()
    # sol_anticausal.set_moments(current_moment=current_moment)
    # e_field_tr = sol_anticausal.compute_e_field(x1, x2, x3, t, h_sym, t_sym)  # np.flip(e_field, axis=-1)

    e_field_norm = np.sum(e_field ** 2, axis=0) ** .5
    # e_field_tr = -np.flip(e_field, axis=-1)
    # r = np.sqrt(x1[:, np.newaxis]**2 + x2[np.newaxis, :]**2)
    plot_data = e_field_norm
    # for index, val in np.ndenumerate(plot_data):
    #     r_2 = x1[index[0]] ** 2 + x2[index[1]] ** 2
    #     if r_2 < (.2 * a) ** 2:
    #         plot_data[index] = 0.
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
        plt.contourf(
            x1, x2,
            plot_data[:, :, 0, index_t].T,
            levels=np.linspace(0, max_data, 26),
            cmap="hsv"
        )
        plt.title(f"{t_i=}")
        if not did_colorbar:
            plt.colorbar()
            did_colorbar = True
        plt.tight_layout()
        # plt.waitforbuttonpress()
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
    parsed = parser.parse_args()
    main(**vars(parsed))

