import numpy as np
import pynoza
import matplotlib.pyplot as plt


def get_scalar_moment(ax, ay, az):
    if ay == ax == 0:
        return [0., 0., 1.1 ** -az]
    return [0., 0., 0.]


def get_h_dict(h, max_order):
    h_dict = dict()
    for az in range(max_order + 1):
        m = get_scalar_moment(0, 0, az)
        h_dict[(0, 0, az)] = [mi * h for mi in m]
    return h_dict


def test_equivalence(plot=False):
    max_order = 4
    t = np.linspace(-3, 3, 1000)
    h = np.exp(- (t * 3) ** 2)
    h_dict = get_h_dict(h, max_order)
    sol = pynoza.Solution(max_order=max_order + 2, wave_speed=1)
    sol.recurse()
    x1 = np.array((10, ))
    x2 = np.array((0, ))
    x3 = np.array((0, ))

    e_field_non_sep = sol.compute_e_field(x1, x2, x3, t, h_dict, delayed=False)

    sol.set_moments(current_moment=get_scalar_moment)

    e_field_sep = sol.compute_e_field(x1, x2, x3, t, h, delayed=False)

    assert np.allclose(e_field_sep, e_field_non_sep, atol=1e-6)

    if plot:
        plt.plot(t, e_field_sep[:, 0, 0, 0, :].T)
        plt.plot(t, e_field_non_sep[:, 0, 0, 0, :].T, "--")
        plt.show()


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("TkAgg")
    test_equivalence(plot=True)
