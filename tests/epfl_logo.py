import pynoza
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import itertools
from test_EPFL_logo import c_j
import sympy


def main():
    gamma_si = 4e-9
    gamma = 1
    c0 = 299792458. * gamma_si

    order = 20
    a = 3

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
    t = np.linspace(1.4, 3, 1)

    def current_moment(ax_, ay, az):
        moment = 0
        if az == 0:
            for x_i, y_i, wi, hi in zip(x1s, x2s, ws, hs):
                moment += c_j(ax_, ay, az, x_i / length_logo * length - d1,
                              y_i / length_logo * length - d2, wi / length_logo * length,
                              hi / length_logo * length) / gamma_si
        return [0, 0, moment]

    n_pts = 64
    x1 = np.linspace(-7, 7, n_pts)
    x2 = np.linspace(-7, 7, n_pts)
    x3 = np.array((0.,))
    t_sym = sympy.Symbol("t", real=True)
    h_sym = sympy.erfc((t_sym - gamma) / gamma)

    sol = pynoza.solution.Solution(max_order=order,
                                   wave_speed=c0)
    sol.recurse()
    sol.set_moments(current_moment=current_moment)
    e_field = sol.compute_e_field(x1, x2, x3, t, h_sym, t_sym, verbose=False)
    e_field_norm = np.sum(e_field ** 2, axis=0) ** .5
    plot_data = np.log10(e_field_norm)
    # for index, val in np.ndenumerate(plot_data):
    #     r_2 = x1[index[0]] ** 2 + x2[index[1]] ** 2
    #     if r_2 < (.1 * a) ** 2:
    #         plot_data[index] = 0.
    max_data = np.max(plot_data)
    plt.figure(
        figsize=(6, 6)
    )
    plt.ion()
    for index_t, t_i in enumerate(t):
        plt.gca().clear()
        plt.contourf(
            x1, x2,
            plot_data[:, :, 0, index_t].T,
            levels=np.linspace(-1, 8, 100),
            cmap="hsv"
        )
        plt.title(f"{t_i=}")
        plt.waitforbuttonpress()
    plt.savefig(f"data/logo_order-{order}.pdf")


if __name__ == "__main__":
    matplotlib.use("TkAgg")
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "Times"
    })
    main()

