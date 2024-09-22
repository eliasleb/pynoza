import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from itertools import product
import inverse_problem
from scipy.interpolate import interp1d
from pynoza.helpers import cache_function_call

global moments_shape, max_order, order_scale, n_tail


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
    # r_km = np.array((1., 5., 10., ))
    # z_km = np.array((0., ))

    t, e_field, h_field, n_t = None, None, None, None
    ind_x = 0
    x, y, z = [], [], []
    for r in r_km:
        for zi in z_km:
            _t, e_r, e_z, h_phi = read_data(
                f"data/lightning_data/20240730/EMF_T={window_us}_r={int(r)}_z={int(zi)}.mat"
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
    y = np.zeros(t_.shape)
    delta_t0 = (np.max(t_) - np.min(t_)) / (h_.size - 1)
    sigma = delta_t0 / 2
    t_max = h_.size / (n_tail + h_.size) * (np.max(t_) - np.min(t_)) + np.min(t_)
    for ind, t0 in enumerate(np.linspace(np.min(t_), t_max, h_.size)):
        y = y + h_[ind] * np.exp(-1/2*(t_ - t0)**2 / sigma**2)
    return y
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
    global max_order, order_scale, moments_shape, n_tail
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

    if plot or plot_recall:
        import matplotlib
        matplotlib.use("TkAgg")

    x, y, z, t, e_field, h_field = read_all_data()

    n = 200
    dt = t[1] - t[0]
    t = np.concatenate((np.linspace(-n * dt, -dt, n) + t[0], t))
    e_field = np.concatenate((np.zeros((3, e_field.shape[1], n)), e_field), axis=-1)
    n_d = 10
    n_t = t.size
    t = t[:n_t//2:n_d]
    e_field = e_field[:, :, :n_t//2:n_d]

    # e_field /= 1e2
    c0 = 3e8
    # t *= c0
    x = x / c0
    y = y / c0
    z = z / c0

    moments_shape = (3, ) + (max_order + 3, ) * 3
    dim_moment = sum(1 for az in range(max_order + 1) if az % 2 == 0)

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
        n_points=n_points,
        find_center=find_center,
        shift=0,
        rescale_at_points=rescale_at_points,
        find_center_ignore_axes=("x", "y"),
        center_scale=center_scale,
        seed=seed,
        # test_indices=()
    )
    current_moment, h, center, e_opt = cache_function_call(
        inverse_problem.inverse_problem,
        **kwargs
    )
    error = np.sqrt(np.sum((e_opt - e_field) ** 2) / np.sum(e_field ** 2))
    if plot_recall:
        print(max_order, seed, n_points, error*100)

        # if error**.5*100 > 31:
        #     raise ValueError()
        plt.figure()
        i_max = h[np.argmax(np.abs(h))]
        plt.plot(t, heidler(t, i_max, tau1=1.8e-6, tau2=95e-6, n=2), "r--")
        plt.plot(t, h, "k-")
        plt.xlabel("Time")

        plt.figure(figsize=(10, 10))
        omin, omax = np.min(e_field), np.max(e_field)
        for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
            plt.subplot(3, 3, i+1)
            plt.plot(t, e_field[:, i, :].T, "r--")
            plt.plot(t, e_opt[:, i, :].T, "k-")
            plt.title(f"{xi*c0/1e3:.1f}, {yi*c0/1e3:.1f}, {zi*c0/1e3:.1f}")
            # plt.ylim(1.1*omin, 1.1*omax)
        plt.tight_layout()

        print(f"{center=}")
        current_moment[-1, 0, 0, :] *= order_scale**np.linspace(0, max_order + 2, max_order + 3)
        inverse_problem.plot_moment(current_moment)
        # plt.figure()
        # max_h = np.max(np.abs(h))
        # if max_h > 0:
        #     plt.plot(t, h / max_h)
        # plt.xlabel("Time (relative)")
        # plt.ylabel("Amplitude (normalized)")
        # plt.title("Current vs time")
        plt.show(block=True)

    return current_moment, h, center, e_opt, error


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
                    current_moment, h, center, e_opt, error = lightning_inverse_problem(**kwargs)
                    errors[seed] = error
                except ValueError:
                    print("Not available")
                    continue
            errors = np.array(errors)
            seed = int(np.argmin(errors))
            kwargs["seed"] = seed
            kwargs["plot_recall"] = True
            current_moment, h, center, e_opt, error = lightning_inverse_problem(**kwargs)


if __name__ == "__main__":
    from_command_line()
    # sweep_results()
