from numpy.random import normal
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from pynoza.inverse_problem import inverse_problem
from pynoza.helpers import cache_function_call
from pynoza.from_mathematica import SPHERICAL_TO_CARTESIAN
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

global moments_shape, max_order, order_scale, dim_moment, n_points, n_z


def heidler(t, max_current, tau1, tau2, n):
    i = max_current / np.exp(-(tau1 / tau2)**n * np.sqrt(n * tau2 / tau1)) * (t / tau1) ** n / (
            1 + (t / tau1)**n) * np.exp(-t / tau2)
    i[t < 0.] = 0.
    return i


def read_data(filename):
    raw_data = loadmat(filename)
    t, e_r, e_z, h_phi = raw_data["time"], raw_data["er"], raw_data["ez"], raw_data["hphi"]
    return t.squeeze(), e_r.squeeze(), e_z.squeeze(), h_phi.squeeze()


# noinspection PyTypeChecker
def read_all_data(window_us=1000, case="MTLL"):
    # r <> x, y <> not used, z <> z
    # r_km = np.array((1., 5., 10.))
    z_km = np.array((0., 2., 4.))
    r_km = np.array((5., 10., ))
    # z_km = np.array((0., ))

    t, e_field, h_field, n_t = None, None, None, None
    ind_x = 0
    x, y, z = [], [], []
    # noinspection PyTypeChecker
    for r in r_km:
        for zi in z_km:
            _t, e_r, e_z, h_phi = read_data(
                f"data/{case}/EMF_T={window_us}_r={int(r)}_z={int(zi)}.mat"
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
        for alpha, coefficient in sph_to_cart.items():
            _current_moment[2, alpha[0], alpha[1], alpha[2]] += moment[ind] * coefficient \
                                                                / order_scale**np.sum(alpha)
        ind += 1

    assert ind == moment.size
    return _current_moment


def get_poly(z, poly):
    # fun = 0. * z
    # for order, coeff in enumerate(poly):
    #     fun += coeff * z ** order
    fun = poly[0] * (np.max(z) - z)
    # plt.clf()
    # plt.plot(z, fun)
    # plt.waitforbuttonpress()
    return fun


def get_current_moment_from_polynomial(poly):
    _current_moment = np.zeros(moments_shape)
    z = np.linspace(0, 4e3/3e8, 100)
    dz = z[1] - z[0]
    fun = get_poly(z, poly)
    for az in range(0, max_order + 1, 2):
        _current_moment[2, 0, 0, az] = 2 * dz * np.sum(z ** az * fun)  # / order_scale**az
    return _current_moment


def get_current_moment(moment):
    _current_moment = np.zeros(moments_shape)
    _current_moment[2, 0, 0, 0] = 1.
    ind = 0
    for az in range(max_order + 1):
        if az % 2 == 1:
            continue
        _current_moment[2, 0, 0, az] = moment[ind] / order_scale**az
        ind += 1

    assert ind == moment.size
    return _current_moment


def get_attenuation(h_):
    global n_z
    z = np.linspace(0, 1, 100)
    attenuation = interp1d(np.linspace(0, 1, n_z + 1), np.concatenate(((1., ), h_[:n_z])),
                           kind="linear", bounds_error=False, fill_value=0.)(z)
    return z, attenuation


def get_h_num_sampled_attenuation(h_, t_):
    global n_z
    n_points_segment = h_.size  # - n_z
    h_dict = dict()
    t_min, t_max = -.3, 5.5
    c = .5
    # steps = np.logspace(np.log10(t_min - t_min + 1), np.log10(t_max - t_min + 1), n_points_segment) + t_min - 1
    h = interp1d(
        np.linspace(t_min, t_max, n_points_segment),
        h_[n_z:],
        kind="cubic",
        fill_value=0.,
        bounds_error=False
    )(t_)
    dt = t_[1] - t_[0]
    step = (t_max - t_min) / (n_points_segment + 1)
    h = gaussian_filter(h, int(step / dt * .5))
    z, attenuation = get_attenuation(h_)
    dz = z[1] - z[0]

    current = np.zeros((t_.size, z.size))
    # plt.clf()
    # plt.plot(z, attenuation)
    # plt.pause(1e-6)
    for i_z, (zi, ai) in enumerate(zip(z, attenuation)):
        current[:, i_z] = interp1d(t_, h, kind="linear", fill_value=0., bounds_error=False)(t_ - zi / c) * ai
    for az in range(0, max_order + 1):
        h_dict[(0, 0, az)] = [
            0 * t_,
            0 * t_,
            dz * np.sum(current * z ** az, axis=1)
        ]
    return h_dict


def get_h_num_full(h_, t_, t_min=0., t_max=1.):
    n_points_segment = h_.size // (max_order // 1 + 1)
    h_dict = dict()
    steps = np.logspace(np.log10(t_min - t_min + 1), np.log10(t_max - t_min + 1), n_points_segment) + t_min - 1
    for az in range(0, max_order + 1, 1):
        h = interp1d(
            steps,
            h_[az // 1 * n_points_segment:(az // 1 + 1) * n_points_segment],
            kind="cubic",
            fill_value=0.,
            bounds_error=False
        )(t_)
        dt = t_[1] - t_[0]
        step = (t_max - t_min) / (n_points_segment + 1)
        h = gaussian_filter(h, int(step / dt * .5))
        h_dict[(0, 0, az)] = [
            0 * t_,
            0 * t_,
            h / order_scale ** az
        ]
    return h_dict


def read_all_data_coupling(x_f, y_f, case, z=10, dx=10, dt=1e-8, n_t=2000):
    filepath = f"../../../lightning_inverse/Campi_Analitici_v2024/Fields_Elias/xF={x_f}_yF={y_f}_P={case}.mat"
    data = loadmat(filepath)
    e_r, e_z, h_phi = data["Er"], data["Ez"], data["Hphi"]
    x = np.arange(0, dx * e_r.shape[0], dx)
    t = np.arange(0, dt * (e_r.shape[1] + 1), dt)[:e_r.shape[1]]
    y = np.zeros(x.shape)
    z = z * np.ones(x.shape)
    x = x - x_f
    y = y - y_f
    phi_rad = np.arctan2(y, x)[:, None]
    e_field = np.stack((
        e_r, np.zeros(e_r.shape), e_z
    ))
    h_field = np.stack((
        -np.sin(phi_rad) * h_phi, np.cos(phi_rad) * h_phi, np.zeros(h_phi.shape)
    ))

    return x, y, z, t, e_field, h_field


def extend_moment(moment, extended_order=10):

    # def exponential_decay(x, a, b):
    #     return a * np.exp(x / b)

    known_orders = np.arange(0, 2 * moment.size, 2)
    rate = np.mean(np.diff(np.log10(np.abs(moment))))
    extended_orders = np.arange(known_orders[-1], extended_order + 1, 2)
    extended_moments = moment[0] * 10**(rate * extended_orders / 2)
    # plt.clf()
    # plt.semilogy(known_orders, np.abs(moment))
    # plt.semilogy(extended_orders, np.abs(extended_moments))
    # plt.waitforbuttonpress()

    return np.concatenate((moment, extended_moments))
    # return extrapolated_moments
    # extended_moments = np.concatenate((moment, extrapolated_moments,))
    # pass


def lightning_inverse_problem(**kwargs):
    global max_order, order_scale, moments_shape, n_points, n_z, dim_moment
    max_order = kwargs.pop("max_order", 0)
    plot = kwargs.pop("plot", False)
    verbose_every = kwargs.pop("verbose_every", 100)
    scale = kwargs.pop("scale", 1e0)
    tol = kwargs.pop("tol", 1e-12)
    n_points = kwargs.pop("n_points", 20)
    find_center = kwargs.pop("find_center", False)
    seed = kwargs.pop("seed", 0)
    order_scale = kwargs.pop("order_scale", 1.)
    center_scale = kwargs.pop("center_scale", 2e3/3e8)
    plot_recall = kwargs.pop("plot_recall", plot)
    rescale_at_points = kwargs.pop("rescale_at_points", False)
    noise_level = kwargs.pop("noise_level", 0.)
    case = kwargs.pop("case")
    raise_error_if_not_cached = kwargs.pop("raise_error_if_not_cached", False)
    study = kwargs.pop("study", "FDTD")
    do_cache = kwargs.pop("do_cache", False)
    z0 = kwargs.pop("z0", 0)

    fit_on_magnetic_field = True

    # if plot or plot_recall:
    #     import matplotlib
    #     matplotlib.use("TkAgg")

    if study == "FDTD":
        x, y, z, t, e_field, h_field = read_all_data(case=case)
        x0, y0 = 0, 0
    elif study == "coupling":
        x0, y0 = 500, 100
        x, y, z, t, e_field, h_field = read_all_data_coupling(x0, y0, "MTLE")
        z = z - z0
        keep = slice(0, -1, 10)
        x, y, z, e_field, h_field = x[keep], y[keep], z[keep], e_field[:, keep, :], h_field[:, keep, :]
    else:
        raise ValueError(f"Unknown value of --study: {study}")

    if fit_on_magnetic_field:
        field = h_field * 4 * np.pi * 1e-7
    else:
        field = e_field

    n = 500
    dt = t[1] - t[0]
    t = np.concatenate((np.linspace(-n * dt, -dt, n) + t[0], t))
    field = np.concatenate((np.zeros((3, field.shape[1], n)), field), axis=-1)
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - 0) ** 2)
    for ind_pos, ri in enumerate(r):
        field[:, ind_pos, ...] = interp1d(t - ri / 3e8, field[:, ind_pos, ...],
                                          bounds_error=False, fill_value=0.)(t)

    # Add noise_e
    if noise_level > 0.:
        np.random.seed(seed)
        noise = normal(size=field.shape)
        noise = noise / np.sqrt(np.sum(noise ** 2)) * np.sqrt(np.sum(field ** 2)) * noise_level
        # plt.hist(noise_e[0, 0, :])
        field = field + noise

    factor = 300.
    x = x / factor
    y = y / factor
    z = z / factor
    t = t * 3e8 / factor
    moments_shape = (3, ) + (max_order + 3, ) * 3
    dim_moment = 0
    test_indices = set(range(0, x.size, 100 // 50))
    t_min = -3
    t_max = 18.
    ignore_tail = 5
    # t_max = np.max(t) - ignore_tail
    # n_z = 2

    bounds = [(-np.inf, 0), ] * (n_points * (max_order // 1 + 1))

    if find_center:
        bounds += [(-np.inf, np.inf), ]

    kwargs = dict(
        order=max_order + 2,
        e_true=field,
        x1=x,
        x2=y,
        x3=z,
        t=t,
        _t_sym=None,
        current_moment_callable=lambda *args: [0., 0., 0.],
        dim_moment=dim_moment,
        h_num=lambda x1, x2: get_h_num_full(x1, x2, t_min=t_min, t_max=t_max),
        plot=plot,
        verbose_every=verbose_every,
        scale=scale,  # shift, scale: 0, 14 -- -1 22
        tol=tol,
        n_points=n_points * (max_order // 1 + 1),
        find_center=find_center,
        shift=0,
        rescale_at_points=rescale_at_points,
        find_center_ignore_axes=("x", "y"),
        center_scale=center_scale,
        seed=seed,
        test_indices=test_indices,
        fit_on_magnetic_field=fit_on_magnetic_field,
        b_true=field,
        delayed=False,
        ignore_tail=ignore_tail,
        return_x_opt=True,
        random_start=False,
        minimize_kwargs=dict(
            bounds=bounds,
        ),
        minimize_options=dict(
            ftol=1e-8,
            gtol=1e-8,
            disp=True,
            maxfun=1_000_000_000_000,
            maxiter=1_000_000_000_000,
            maxls=(n_points * (max_order // 1 + 1)) * 5
        )
    )
    print(f"{field.shape=}, {len(test_indices)=}")
    if do_cache:
        current_moment, h, center, field_opt, x_opt = cache_function_call(
            inverse_problem,
            raise_error_if_not_cached=raise_error_if_not_cached,
            **kwargs
        )
    else:
        current_moment, h, center, field_opt, x_opt = inverse_problem(
            **kwargs
        )

    if plot_recall:
        plt.figure()
        plt.stairs(x_opt)
        plt.title("x opt")

    train_indices = set(range(x.size)) - set(test_indices)
    keep = (t <= np.max(t) - ignore_tail)
    train_indices, test_indices = list(sorted(train_indices)), list(sorted(test_indices))
    error_train = np.sqrt(np.sum((field_opt[..., keep][:, train_indices, :]
                                  - field[..., keep][:, train_indices, :]) ** 2)
                          / np.sum(field[..., keep][:, train_indices, :] ** 2))
    try:
        error_test = np.sqrt(np.sum((field_opt[..., keep][:, test_indices, :]
                                     - field[..., keep][:, test_indices, :]) ** 2)
                              / np.sum(field[..., keep][:, test_indices, :] ** 2))
    except FloatingPointError:
        error_test = 1.
    if plot_recall:
        t = t * factor / 3e8 * 1e6
        t_min = np.min(t)
        t_max *= factor / 3e8 * 1e6
        print(f"{max_order=}, {seed=}, {n_points=}, train={error_train*100:.1f}, test={error_test*100:.1f}")
        if isinstance(center, float):
            print(f"{center*4e3=}")

        if isinstance(h, dict):
            h = np.array([h[(0, 0, az)] for az in range(0, max_order + 1, 1)])
            arg_max = np.argmax(np.abs(h), axis=-1)
            h_max = h[0, 0, arg_max[0]]
        else:
            arg_max = np.argmax(np.abs(h))
            h_max = h[arg_max]

        path = "../../../lightning_inverse/figs"

        plt.figure()
        # heidler_scaled = heidler(t, h_max/.85, tau1=1.8e-6*factor, tau2=95e-6*factor, n=2)
        # plt.plot(t - t[np.argmax(np.abs(heidler_scaled))] + t[arg_max], heidler_scaled, "r--", label="Heidler")
        if len(h.shape) > 1 and h.shape[0] > 1:
            for f_ind in range(h.shape[0]):
                plt.plot(t, h[f_ind, 2], color=plt.get_cmap("jet")(f_ind/(h.shape[0]-1)), label=f"{f_ind}")
        else:
            if len(h.shape) == 3:
                plt.plot(t, h[:, 2, :].T, "k", label="h")
            else:
                plt.plot(t, h, "k", label="h")

        plt.legend()
        plt.xlabel("Time (us)")
        plt.tight_layout()
        plt.savefig(f"{path}/opt_h_{case}.pdf")

        plt.show(block=True)

    return current_moment, h, center, field_opt, error_train, error_test


def from_command_line():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--max_order", type=int, required=True)
    parser.add_argument("--verbose_every", type=int, default=100)
    parser.add_argument("--scale", type=float, default=1.)
    parser.add_argument("--plot", type=bool, default=False)
    parser.add_argument("--n_points", type=int, default=20)
    parser.add_argument("--order_scale", type=float, default=1.)
    parser.add_argument("--center_scale", type=float, default=2e3/3e8)
    parser.add_argument("--find_center", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot_recall", type=bool, default=False)
    parser.add_argument("--rescale_at_points", type=bool, default=False)
    parser.add_argument("--noise_level", type=float, default=0.)
    parser.add_argument("--case", type=str, default="MTLL")
    parser.add_argument("--study", type=str, default="FDTD")
    parser.add_argument("--do_cache", type=int, default=False)
    parser.add_argument("--z0", type=int, default=0)

    parsed = parser.parse_args()
    lightning_inverse_problem(**vars(parsed))


def sweep_results():
    all_orders = (2, 4, 6, )
    all_n_points = range(20, 51, 1)
    seeds = (11, )
    for case in ("MTLL", "MTLE", "QUAD", "TL", ):
        errors = np.ones((len(all_orders), len(all_n_points), len(seeds),)) * np.nan
        for ind_order, order in enumerate(all_orders):
            for ind_n_points, _n_points in enumerate(all_n_points):
                for ind_seed, seed in enumerate(seeds):
                    kwargs = dict(
                        max_order=order,
                        plot=False,
                        plot_recall=False,
                        scale=1e9,
                        n_points=_n_points,
                        seed=seed,
                        case=case,
                        order_scale=2,
                        raise_error_if_not_cached=True
                    )
                    print(f"{case=}, {order=}, {_n_points=}")
                    try:
                        current_moment, h, center, e_opt, train_error, test_error = lightning_inverse_problem(**kwargs)
                        errors[ind_order, ind_n_points, ind_seed] = test_error
                    except FileNotFoundError:
                        print("Not found, ignoring")
        ind_order, ind_n_points, ind_seed = np.unravel_index(np.nanargmin(errors), errors.shape)
        if len(all_orders) > 1:
            plt.figure()
            plt.contourf(
                all_orders, all_n_points, np.log10(np.min(errors, axis=-1)).T, cmap="jet", levels=31
            )
            plt.colorbar()

        kwargs["seed"] = seeds[ind_seed]
        kwargs["max_order"] = all_orders[ind_order]
        kwargs["n_points"] = all_n_points[ind_n_points]
        kwargs["plot_recall"] = True
        current_moment, h, center, e_opt, train_error, test_error = lightning_inverse_problem(**kwargs)


if __name__ == "__main__":
    import matplotlib
    try:
        matplotlib.use("macosx")
    except ModuleNotFoundError:
        pass
    from_command_line()
    # sweep_results()
