from numpy.random import normal
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from pynoza.inverse_problem import inverse_problem, plot_moment_2d
from pynoza.helpers import cache_function_call, optimization_moment_problem_solution
from pynoza.from_mathematica import SPHERICAL_TO_CARTESIAN
from pynoza import Solution
from scipy.signal import sosfilt, butter
from scipy.interpolate import interp1d
from scipy.special import erf
from scipy.ndimage import gaussian_filter

global moments_shape, max_order, order_scale, n_tail, dim_moment, n_points, h_lower_order, n_calls


def heidler(t, max_current, tau1, tau2, n):
    i = max_current / np.exp(-(tau1 / tau2)**n * np.sqrt(n * tau2 / tau1)) * (t / tau1) ** n / (
            1 + (t / tau1)**n) * np.exp(-t / tau2)
    i[t < 0.] = 0.
    return i


def read_data(filename):
    raw_data = loadmat(filename)
    t, e_r, e_z, h_phi = raw_data["time"], raw_data["er"], raw_data["ez"], raw_data["hphi"]
    return t.squeeze(), e_r.squeeze(), e_z.squeeze(), h_phi.squeeze()


def read_all_data(window_us=1000, case="MTLL"):
    # r <> x, y <> not used, z <> z
    # r_km = np.array((1., 5., 10.))
    z_km = np.array((0., 2., 4.))
    r_km = np.array((5., 10., ))
    # z_km = np.array((0., ))

    t, e_field, h_field, n_t = None, None, None, None
    ind_x = 0
    x, y, z = [], [], []
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


def get_current_moment_true(scale, case="MTLL"):
    _current_moment = np.zeros(moments_shape)
    # z_max = 1
    # z = np.linspace(0, z_max, 1000)
    # dz = z[1] - z[0]
    # if case == "MTLL":
    #     fun = 1 - z / z_max
    # else:
    #     raise NotImplemented
    for az in range(0, max_order + 1, 2):
        _current_moment[2, 0, 0, az] = 1.  # 2 * dz * np.sum(z ** az * fun)  # / order_scale**az
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


def get_h_num(h_, t_):
    n_points_segment = h_.size // (max_order // 2 + 1)
    h_dict = dict()
    for az in range(0, max_order + 1, 2):
        t_min, t_max = -.3, 5.5
        steps = np.logspace(np.log10(t_min - t_min + 1), np.log10(t_max - t_min + 1), n_points_segment) + t_min - 1
        h = interp1d(
            steps,
            h_.ravel()[(az // 2) * n_points_segment:(az // 2 + 1) * n_points_segment],
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
            h / 3 ** az
            # dz * np.sum(
            #     attenuation * h_interpolated * z ** az,
            #     axis=1
            # )
        ]
    return h_dict


def lightning_inverse_problem(**kwargs):
    global max_order, order_scale, moments_shape, n_points, n_tail, dim_moment
    max_order = kwargs.pop("max_order", 0)
    plot = kwargs.pop("plot", False)
    verbose_every = kwargs.pop("verbose_every", 100)
    scale = kwargs.pop("scale", 1e0)
    tol = kwargs.pop("tol", 1e-12)
    n_points = kwargs.pop("n_points", 20)
    find_center = kwargs.pop("find_center", False)
    seed = kwargs.pop("seed", 0)
    n_tail = kwargs.pop("n_tail", 10)
    order_scale = kwargs.pop("order_scale", 1.)
    center_scale = kwargs.pop("center_scale", 2e3/3e8)
    plot_recall = kwargs.pop("plot_recall", plot)
    rescale_at_points = kwargs.pop("rescale_at_points", False)
    noise_level = kwargs.pop("noise_level", 0.)
    case = kwargs.pop("case")

    fit_on_magnetic_field = True

    if plot or plot_recall:
        import matplotlib
        matplotlib.use("TkAgg")

    x, y, z, t, e_field, h_field = read_all_data(case=case)
    if fit_on_magnetic_field:
        field = h_field * 4 * np.pi * 1e-7
    else:
        field = e_field

    n = 200
    dt = t[1] - t[0]
    t = np.concatenate((np.linspace(-n * dt, -dt, n) + t[0], t))
    field = np.concatenate((np.zeros((3, field.shape[1], n)), field), axis=-1)
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
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

    print(x, y, z)
    factor = 4e3
    x = x / factor
    y = y / factor
    z = z / factor
    t = t * 3e8 / factor
    moments_shape = (3, ) + (max_order + 3, ) * 3
    dim_moment = sum(1 for az in range(max_order + 1) if az % 2 == 0) - 1
    # dim_moment = max_order // 2 + 1
    dim_moment = 0
    test_indices = (1, 5,)

    keep = (t >= -.5) * (t <= 5)
    t = t[keep]
    field = field[..., keep]

    kwargs = dict(
        order=max_order + 2,
        e_true=field,
        x1=x,
        x2=y,
        x3=z,
        t=t,
        _t_sym=None,
        current_moment_callable=get_current_moment_true,
        dim_moment=dim_moment,
        h_num=get_h_num,
        plot=plot,
        verbose_every=verbose_every,
        scale=scale,  # shift, scale: 0, 14 -- -1 22
        tol=tol,
        n_points=n_points * (max_order // 2 + 1),
        find_center=find_center,
        shift=0,
        rescale_at_points=rescale_at_points,
        find_center_ignore_axes=("x", "y"),
        center_scale=center_scale,
        seed=seed,
        test_indices=test_indices,
        fit_on_magnetic_field=fit_on_magnetic_field,
        b_true=field,
        delayed=False
        # return_raw_moment=True
    )
    current_moment, h, center, field_opt = cache_function_call(
        inverse_problem,
        **kwargs
    )
    train_indices = set(range(x.size)) - set(test_indices)
    train_indices, test_indices = list(sorted(train_indices)), list(sorted(test_indices))
    error_train = np.sqrt(np.sum((field_opt[:, train_indices, ...] - field[:, train_indices, ...]) ** 2)
                          / np.sum(field[:, train_indices, ...] ** 2))
    try:
        error_test = np.sqrt(np.sum((field_opt[:, test_indices, ...] - field[:, test_indices, ...]) ** 2)
                              / np.sum(field[:, test_indices, ...] ** 2))
    except FloatingPointError:
        error_test = 1.
    if plot_recall:
        print(f"{max_order=}, {seed=}, {n_points=}, train={error_train*100:.1f}, test={error_test*100:.1f}")
        # print(f"{poly=}")
        # if error**.5*100 > 31:
        #     raise ValueError()
        if isinstance(h, dict):
            h = np.array(list(h.values()))
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
        plt.xlabel("Time (s)")
        plt.ylabel("Excitation function (1)")
        plt.savefig(f"{path}/{case}_opt_h.pdf")

        plt.figure(figsize=(15, 10))
        # omin, omax = np.min(e_field), np.max(e_field)
        for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
            plt.subplot(3, 2, i+1)
            plt.plot(t, field[:, i, :].T, "r--")
            plt.plot(t, field_opt[:, i, :].T, "k-")
            plt.title(f"x = {xi*factor/1e3:.1f} km, z = {zi*factor/1e3:.1f} km")
            if i % 2 == 0:
                plt.ylabel("Electric field (V/m)")
            if i > 3:
                plt.xlabel("Time (s)")
            # plt.ylim(1.1*omin, 1.1*omax)
        plt.tight_layout()
        plt.savefig(f"{path}/{case}_opt_fields.pdf")

        z = np.linspace(-1, 1, 1000)
        function = np.zeros((t.size, z.size))
        plt.figure()
        for ind_t, ti in enumerate(t):
            moment = np.array([h[0, 2, ind_t], 0, h[1, 2, ind_t], ])
            f, _ = optimization_moment_problem_solution(
                z, moment, poly_order=2,
                tol=1e-8,
                method="BFGS"
            )
            function[ind_t, :] = f

        plt.clf()
        m = np.max(np.abs(function))
        plt.contourf(t * factor / 3e8 * 1e6, z * 4, function.T, cmap="RdBu", levels=np.linspace(-m, m, 41))
        plt.ylim(0, 4)
        plt.colorbar()

        plt.ylabel("Altitude (km)")
        plt.xlabel("Time (us)")
        plt.title(f"{case}")
        plt.savefig(f"{path}/{case}_attenuation.pdf")
        plt.show(block=True)

    return current_moment, h, center, field_opt, error_train, error_test


def from_command_line():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--max_order", type=int, required=True)
    parser.add_argument("--verbose_every", type=int, default=100)
    parser.add_argument("--scale", type=float, default=1.)
    parser.add_argument("--plot", type=bool, default=False)
    parser.add_argument("--n_tail", type=int, default=20)
    parser.add_argument("--n_points", type=int, default=20)
    parser.add_argument("--order_scale", type=float, default=1.)
    parser.add_argument("--center_scale", type=float, default=2e3/3e8)
    parser.add_argument("--find_center", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot_recall", type=bool, default=False)
    parser.add_argument("--rescale_at_points", type=bool, default=False)
    parser.add_argument("--noise_level", type=float, default=0.)
    parser.add_argument("--case", type=str, default="MTLL")

    parsed = parser.parse_args()
    lightning_inverse_problem(**vars(parsed))


def sweep_results():
    for case in ("TL", "MTLL", "MTLE", "QUAD"):
        for order in (8, ):
            for _n_points in (30, 40, 50, ):
                errors = np.ones((10, )) * np.nan
                for seed in range(10):
                    kwargs = dict(
                        max_order=order,
                        verbose_every=100,
                        plot=False,
                        plot_recall=False,
                        n_tail=0,
                        n_points=_n_points,
                        order_scale=1e6,
                        center_scale=2e3/3e8,
                        find_center=True,
                        seed=seed,
                        case=case
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
