from pynoza.inverse_problem import inverse_problem
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from pynoza import Solution
from pynoza.helpers import cache_function_call, optimization_moment_problem_solution
import pytest


def true_current_density(z, size=1., fun_type="TL"):
    if fun_type == "MTLL":
        return (1 - np.abs(z/size)) * (np.abs(z) < size)
    elif fun_type == "QUAD":
        return (1 - z**2) * (np.abs(z) < size)
    elif fun_type == "MTLE":
        return np.exp(-2 * np.abs(z)) * (np.abs(z) < size)
    return np.ones(z.shape)


def get_true_current_moment(ax, ay, az, z_j, j_z):
    dz_j = z_j[1] - z_j[0]
    if ax == ay == 0:
        return [0., 0., np.sum(z_j**az * j_z) * dz_j]

    return [0., 0., 0.]


def get_current_moment(h, max_order=0):
    ind = 0
    moment = np.zeros(shape=(3, ) + (max_order + 3, ) * 3)
    for az in range(max_order + 1):
        moment[2, 0, 0, ind] = h[ind]
        ind += 1
    return moment


def get_h_num(h_, t_, n_points=30):
    n_tail = n_points * 6 // 30
    delta_t0 = (np.max(t_) - np.min(t_)) / (h_.size - 1)
    sigma = delta_t0 / 1
    t_max = h_.size / (n_tail + h_.size) * (np.max(t_) - np.min(t_)) + np.min(t_)
    ind = 0
    y = np.zeros(t_.shape)
    for t0 in np.linspace(np.min(t_), t_max, n_points):
        y = y + h_[ind] * np.exp(-1 / 2 * (t_ - t0) ** 2 / sigma ** 2)
        ind += 1
    return y - y[0]


@pytest.mark.parametrize("fun_type", (
    "TL",
    "MTLL",
    "MTLE",
    "QUAD"
))
def test_em_j_reconstruction(fun_type, plot=False, max_l2_error=1e-2):
    size = 1.
    z_j = np.linspace(-size, size, 10000)
    j_z = true_current_density(z_j, size=size, fun_type=fun_type)

    t = np.linspace(-3, 5, 100)
    t0 = 1
    h = np.exp(-(t / t0) ** 2) * (4 * (t / t0)**2 - 2)

    # if plot:
    #     plt.figure()
    #     plt.plot([get_true_current_moment(0, 0, az, z_j, j_z) for az in range(10)])

    r = 1.
    theta = np.linspace(0, np.pi, 4)
    phi = np.linspace(0, 2 * np.pi, 8)
    x, y, z = [], [], []
    for theta_i, phi_i in product(theta, phi):
        x.append(r * np.sin(theta_i) * np.cos(phi_i))
        y.append(r * np.sin(theta_i) * np.sin(phi_i))
        z.append(r * np.cos(theta_i))
    x, y, z = np.array(x), np.array(y), np.array(z)

    max_order = 8

    direct_problem = Solution(
        max_order=max_order+2
    )
    direct_problem.recurse()
    direct_problem.set_moments(
        current_moment=lambda ax, ay, az: get_true_current_moment(ax, ay, az, z_j, j_z)
    )
    e_field_true = direct_problem.compute_e_field(
        x, y, z, t, h, None,
        compute_grid=False
    )

    n_points = 30
    args = (
        max_order + 2,
        e_field_true,
        x, y, z, t, None,
        lambda hi: get_current_moment(hi, max_order),
        max_order + 1,
    )
    kwargs = dict(
        plot=plot,
        h_num=lambda a, b: get_h_num(a, b, n_points=n_points),
        n_points=n_points,
        scale=1,
        verbose_every=10,
        tol=1e-8,
        find_center=False
    )

    moment_opt, h_opt, center, field_opt = cache_function_call(
        inverse_problem,
        *args,
        **kwargs
    )
    true_moment = np.array([get_true_current_moment(0, 0, az, z_j, j_z)[2] for az in range(max_order+1)])
    reconstructed_moment = moment_opt[2, 0, 0, :-2]
    true_moment = true_moment * np.max(np.abs(h)) / 2.
    reconstructed_moment = -reconstructed_moment * h_opt[np.argmax(np.abs(h_opt))] / 2.
    print(f"True moments: {true_moment}")
    print(f"Reconstructed: {reconstructed_moment}")

    order = 10
    if order > true_moment.size:
        x1 = np.concatenate((true_moment, np.zeros((order - true_moment.size))))
        x2 = np.concatenate((reconstructed_moment, np.zeros((order - reconstructed_moment.size))))
    else:
        x1 = true_moment[:order]
        x2 = reconstructed_moment[:order]
    opt_kwargs = dict(
        method="BFGS",
        tol=1e-17
    )
    order = 2
    y_true, _ = optimization_moment_problem_solution(z_j, x1, poly_order=order, **opt_kwargs)
    y_opt, _ = optimization_moment_problem_solution(z_j, x2, poly_order=order, **opt_kwargs)

    l2_error = np.sqrt(np.sum((y_opt - y_true) ** 2)) / np.sqrt(np.sum(y_true ** 2))
    print(f"{l2_error=}")

    assert l2_error < max_l2_error

    if plot:
        plt.figure()
        plt.plot(z_j, y_true)
        plt.plot(z_j, y_opt)
        plt.plot(z_j, j_z, "r--")
        plt.title(f"{fun_type}, {order=}")

        # plt.figure()
        # plt.title(f"{fun_type}")
        # plt.stairs(true_moment)
        # plt.stairs(reconstructed_moment)


def get_function_reconstruction(z, fun, max_order=10, plot=False, l2_max=np.inf):
    moments = []
    dz = z[1] - z[0]
    for order in range(0, max_order + 1):
        moments.append(
            np.sum(z ** order * fun) * dz
        )
    moments = np.array(moments)

    y, residual = optimization_moment_problem_solution(
        z, moments, poly_order=2,
        method="bfgs",
        tol=1e-14
    )

    # plt.stairs(np.abs(moments))

    l2 = np.sum((y - fun)**2)**.5 / np.sum(fun**2)**.5
    assert l2 < l2_max
    print(f"{l2=}")
    if plot:
        plt.figure()
        plt.plot(z, fun)
        plt.plot(z, y)
        plt.title(f"{max_order=}, {residual=}")


def test_function_reconstruction(plot=False):
    z = np.linspace(-1, 1, 1000)

    functions = [
        1 - np.abs(z),
        1 + 0 * z,
        1 - z**2,
        np.exp(-2 * np.abs(z)),
    ]
    for fun in functions:
        get_function_reconstruction(z, fun, max_order=10, plot=plot, l2_max=2.6e-2)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("TkAgg")

    plot = True
    test_function_reconstruction(plot=plot)
    l2_max = 1.4e-2
    # test_em_j_reconstruction(plot=plot, fun_type="TL", max_l2_error=l2_max)
    # test_em_j_reconstruction(plot=plot, fun_type="MTLL", max_l2_error=l2_max)
    test_em_j_reconstruction(plot=plot, fun_type="MTLE", max_l2_error=l2_max)
    # test_em_j_reconstruction(plot=plot, fun_type="QUAD", max_l2_error=l2_max)

    plt.show(block=True)
