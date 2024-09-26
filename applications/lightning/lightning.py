from numpy.random import normal
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from itertools import product
from pynoza.inverse_problem import inverse_problem
from scipy.interpolate import interp1d
from pynoza.helpers import cache_function_call
from pynoza.from_mathematica import SPHERICAL_TO_CARTESIAN
global moments_shape, max_order, order_scale, n_tail, dim_moment, n_points


def heidler(t, max_current, tau1, tau2, n):
    i = max_current * (t/tau1)**n / (1 + (t/tau1)**n) * np.exp(-t/tau2)
    i[t < 0.] = 0.
    return i


def read_data(filename):
    raw_data = loadmat(filename)
    t, e_r, e_z, h_phi = raw_data["time"], raw_data["er"], raw_data["ez"], raw_data["hphi"]
    return t.squeeze(), e_r.squeeze(), e_z.squeeze(), h_phi.squeeze()


def read_all_data(window_us=1000):
    # r <> x, y <> not used, z <> z
    r_km = np.array((1., 5., 10.))
    z_km = np.array((0., 2., 4.))
    r_km = np.array((5., 10., ))
    # z_km = np.array((0., ))

    t, e_field, h_field, n_t = None, None, None, None
    ind_x = 0
    x, y, z = [], [], []
    for r in r_km:
        for zi in z_km:
            _t, e_r, e_z, h_phi = read_data(
                f"data/20240730/EMF_T={window_us}_r={int(r)}_z={int(zi)}.mat"
            )
            if e_field is None:
                e_field = np.zeros((3, r_km.size * z_km.size, _t.size))
                h_field = e_field.copy()
                t = _t
                n_t = t.size
            e_field[:, ind_x, :] = np.array((
                e_r[:n_t], np.zeros((n_t, )), e_z[:n_t]
            ))
            h_field[:, ind_x, :] = np.array((
                np.zeros((n_t, )), h_phi[:n_t], np.zeros((n_t, ))
            ))
            x.append(r)
            y.append(0)
            z.append(zi)
            ind_x += 1
    x, y, z = np.array(x) * 1e3, np.array(y), np.array(z) * 1e3
    return x, y, z, t, e_field, h_field


def re_or_im(x):
    if np.abs(np.real(x)) > 1e-18:
        return np.real(x)
    return np.imag(x)


def get_current_moment_spherical(moment):
    _current_moment = np.zeros(moments_shape)
    ind = 0
    for l in range(0, max_order + 1, 2):
        sph_to_cart = SPHERICAL_TO_CARTESIAN[(l, 0)]
        for alpha in sph_to_cart:
            _current_moment[2, alpha[0], alpha[1], alpha[2]] += moment[ind] * re_or_im(sph_to_cart[alpha]) \
                                                                / order_scale**np.sum(alpha)
        ind += 1

    assert ind == moment.size
    return _current_moment


def get_current_moment(moment):
    _current_moment = np.zeros(moments_shape)
    ind = 0
    for az in range(max_order + 1):
        if az % 2 == 1:
            continue
        _current_moment[2, 0, 0, az] = moment[ind] / order_scale**az
        ind += 1

    assert ind == moment.size
    return _current_moment


def get_h_num(h_, t_):
    h_dict = dict()
    delta_t0 = (np.max(t_) - np.min(t_)) / (h_.size - 1)
    sigma = delta_t0 / 2
    t_max = h_.size / (n_tail + h_.size) * (np.max(t_) - np.min(t_)) + np.min(t_)
    ind = 0
    # for az in range(max_order + 1):
    #     if az % 2 == 1:
    #         continue
    y = np.zeros(t_.shape)
    # plt.figure()
    for t0 in np.linspace(np.min(t_), t_max, n_points):
        y = y + h_[ind] * np.exp(-1 / 2 * (t_ - t0) ** 2 / sigma ** 2)
        # plt.plot(t_, np.exp(-1 / 2 * (t_ - t0) ** 2 / sigma ** 2), "k--")
        ind += 1
        # h_dict[(0, 0, az)] = y - y[0]
    # plt.xlabel("Normalized time")
    # plt.title("Current basis functions")
    # plt.tight_layout()
    # plt.show()
    return y - y[0]
    #
    # if n_tail > 0:
    #     h_interpolated = interp1d(
    #         np.linspace(np.min(t_), np.max(t_), 1 + h_.size + n_tail),
    #         np.concatenate((np.array((0,)), h_.ravel(), np.zeros((n_tail, )))),
    #         kind="cubic"
    #     )(t_)
    # else:
    #     h_interpolated = interp1d(
    #         np.linspace(np.min(t_), np.max(t_), 1 + h_.size),
    #         np.concatenate((np.array((0,)), h_.ravel())),
    #         kind="cubic"
    #     )(t_)
    # return h_interpolated


def lightning_inverse_problem(**kwargs):
    global max_order, order_scale, moments_shape, n_points, n_tail, dim_moment
    max_order = kwargs.pop("max_order", 0)
    plot = kwargs.pop("plot", False)
    verbose_every = kwargs.pop("verbose_every", 100)
    scale = kwargs.pop("scale", 1e0)
    tol = kwargs.pop("tol", 1e-8)
    n_points = kwargs.pop("n_points", 20)
    find_center = kwargs.pop("find_center", False)
    seed = kwargs.pop("seed", 0)
    n_tail = kwargs.pop("n_tail", 10)
    order_scale = kwargs.pop("order_scale", 1.)
    center_scale = kwargs.pop("center_scale", 2e3/3e8)
    plot_recall = kwargs.pop("plot_recall", plot)
    rescale_at_points = kwargs.pop("rescale_at_points", False)
    noise_level = kwargs.pop("noise_level", 0.)

    if plot or plot_recall:
        import matplotlib
        matplotlib.use("TkAgg")

    x, y, z, t, e_field, h_field = read_all_data()

    n = 200
    dt = t[1] - t[0]
    t = np.concatenate((np.linspace(-n * dt, -dt, n) + t[0], t))
    e_field = np.concatenate((np.zeros((3, e_field.shape[1], n)), e_field), axis=-1)
    n_d = 1
    n_t = t.size
    t = t[:n_t//2:n_d]
    e_field = e_field[:, :, :n_t//2:n_d]

    # Add noise
    if noise_level > 0.:
        np.random.seed(seed)
        noise = normal(size=e_field.shape)
        noise = noise / np.sqrt(np.sum(noise**2)) * np.sqrt(np.sum(e_field**2)) * noise_level
        plt.hist(noise[0, 0, :])
        e_field = e_field + noise

    # e_field /= 1e2
    c0 = 3e8
    # t *= c0
    print(x, y, z)
    x = x / c0
    y = y / c0
    z = z / c0

    moments_shape = (3, ) + (max_order + 3, ) * 3
    dim_moment = sum(1 for az in range(max_order + 1) if az % 2 == 0)
    # dim_moment = max_order // 2 + 1
    test_indices = (1, 5,)

    kwargs = dict(
        order=max_order + 2,
        e_true=e_field,
        x1=x,
        x2=y,
        x3=z,
        t=t,
        _t_sym=None,
        current_moment_callable=get_current_moment,
        dim_moment=dim_moment,
        h_num=get_h_num,
        plot=plot,
        verbose_every=verbose_every,
        scale=scale,  # shift, scale: 0, 14 -- -1 22
        tol=tol,
        n_points=n_points * 1,
        find_center=find_center,
        shift=0,
        rescale_at_points=rescale_at_points,
        find_center_ignore_axes=("x", "y"),
        center_scale=center_scale,
        seed=seed,
        test_indices=test_indices
    )
    current_moment, h, center, e_opt = cache_function_call(
        inverse_problem,
        **kwargs
    )
    train_indices = set(range(x.size)) - set(test_indices)
    train_indices, test_indices = list(sorted(train_indices)), list(sorted(test_indices))
    error_train = np.sqrt(np.sum((e_opt[:, train_indices, ...] - e_field[:, train_indices, ...]) ** 2)
                          / np.sum(e_field[:, train_indices, ...] ** 2))
    error_test = np.sqrt(np.sum((e_opt[:, test_indices, ...] - e_field[:, test_indices, ...]) ** 2)
                          / np.sum(e_field[:, test_indices, ...] ** 2))

    if plot_recall:
        print(f"{max_order=}, {seed=}, {n_points=}, train={error_train*100:.1f}, test={error_test*100:.1f}")

        # if error**.5*100 > 31:
        #     raise ValueError()
        if isinstance(h, dict):
            h = np.array(list(h.values()))
            arg_max = np.argmax(np.abs(h), axis=-1)
            h_max = h[0, arg_max[0]]
        else:
            arg_max = np.argmax(np.abs(h))
            h_max = h[arg_max]

        path = "../../lightning_inverse/figs"

        plt.figure()
        heidler_scaled = heidler(t, h_max/.85, tau1=1.8e-6, tau2=95e-6, n=2)
        plt.plot(t - t[np.argmax(np.abs(heidler_scaled))] + t[arg_max], heidler_scaled, "r--", label="Heidler")
        if len(h.shape) > 1 and h.shape[0] > 1:
            for f_ind in range(h.shape[0]):
                plt.plot(t, h[f_ind, :], color=plt.get_cmap("jet")(f_ind/(h.shape[0]-1)), label=f"{f_ind}")
        else:
            plt.plot(t, h.T, "k", label="h")
        plt.legend()
        plt.xlabel("Normalized time")
        plt.ylabel("Excitation function (1)")
        plt.savefig(f"{path}/opt_h.pdf")

        plt.figure(figsize=(15, 10))
        omin, omax = np.min(e_field), np.max(e_field)
        for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
            plt.subplot(3, 2, i+1)
            plt.plot(t, e_field[:, i, :].T, "r--")
            plt.plot(t, e_opt[:, i, :].T, "k-")
            plt.title(f"x = {xi*c0/1e3:.1f} km, z = {zi*c0/1e3:.1f} km")
            if i % 2 == 0:
                plt.ylabel("Electric field (V/m)")
            if i > 3:
                plt.xlabel("Normalized time")
            # plt.ylim(1.1*omin, 1.1*omax)
        plt.tight_layout()
        plt.savefig(f"{path}/opt_fields.pdf")

        print(f"{center=}")
        current_moment[-1, 0, 0, :] *= order_scale**np.linspace(0, max_order + 2, max_order + 3)
        inverse_problem.plot_moment_2d(current_moment)
        plt.savefig(f"{path}/opt_moments.pdf")
        # plt.figure()
        # max_h = np.max(np.abs(h))
        # if max_h > 0:
        #     plt.plot(t, h / max_h)
        # plt.xlabel("Time (relative)")
        # plt.ylabel("Amplitude (normalized)")
        # plt.title("Current vs time")
        plt.show(block=True)

    return current_moment, h, center, e_opt, error_train, error_test


def from_command_line():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--max_order", type=int, required=True)
    parser.add_argument("--verbose_every", type=int, default=100)

    parser.add_argument("--plot", type=bool, default=False)
    parser.add_argument("--n_tail", type=int, default=20)
    parser.add_argument("--n_points", type=int, default=20)
    parser.add_argument("--order_scale", type=float, default=1e6)
    parser.add_argument("--center_scale", type=float, default=2e3/3e8)
    parser.add_argument("--find_center", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot_recall", type=bool, default=False)
    parser.add_argument("--rescale_at_points", type=bool, default=False)
    parser.add_argument("--noise_level", type=float, default=0.)

    parsed = parser.parse_args()
    lightning_inverse_problem(**vars(parsed))


def sweep_results():
    for order in range(2, 10, 2):
        for n_points in range(40, 150, 10):
            errors = np.ones((10, )) * np.nan
            for seed in range(10):
                kwargs = dict(
                    max_order=order,
                    verbose_every=100,
                    plot=False,
                    plot_recall=False,
                    n_tail=n_points // 10,
                    n_points=n_points,
                    order_scale=1e6,
                    center_scale=2e3/3e8,
                    find_center=True,
                    seed=seed
                )
                try:
                    current_moment, h, center, e_opt, train_error, test_error = lightning_inverse_problem(**kwargs)
                    errors[seed] = test_error
                except ValueError:
                    print("Not available")
                    continue
            errors = np.array(errors)
            seed = int(np.argmin(errors))
            kwargs["seed"] = seed
            kwargs["plot_recall"] = True
            current_moment, h, center, e_opt, train_error, test_error = lightning_inverse_problem(**kwargs)


if __name__ == "__main__":
    from_command_line()
    # sweep_results()
