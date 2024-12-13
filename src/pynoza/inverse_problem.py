import pynoza
import numpy as np
import scipy.optimize
import scipy.interpolate
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from scipy.optimize import dual_annealing


def complement(*args):
    if len(args) == 1:
        i = args[0]
        if i == 0:
            return 1, 2
        elif i == 1:
            return 0, 2
        else:
            return 0, 1
    elif len(args) == 2:
        i, j = args
        if i == j:
            raise ValueError('i and j must be different')
        if {i, j} == {0, 1}:
            return 2
        elif {i, j} == {0, 2}:
            return 1
        else:
            return 0


def plot_moment_2d(moment):
    plt.figure(figsize=(15, 8))
    max_order = moment.shape[1] - 3
    m_max = np.abs(moment).max()
    cmap = plt.get_cmap("RdBu")
    for dim, dim_name in enumerate(("x", "y", "z")):
        for order in range(max_order + 1):
            fig_num = order + (max_order + 1) * dim + 1
            plt.subplot(3, max_order + 1, fig_num)
            if dim == 0:
                plt.title(f"{order}")
            if fig_num % (max_order + 1) == 1:
                plt.ylabel(dim_name)
            for ax in range(order + 1):
                for ay in range(order + 1 - ax):
                    az = order - ax - ay
                    plt.scatter(ax, ay, color=cmap(moment[dim, ax, ay, az]/m_max/2+.5))


def plot_moment(moment):
    moment = moment.swapaxes(0, 1).swapaxes(1, 2).swapaxes(2, 3)
    fig = plt.figure()
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    axes = [ax1, ax2, ax3]

    order = moment.shape[1]
    x1 = np.arange(order)
    x2 = np.arange(order)
    x3 = np.arange(order)
    _x1, _x2, _x3 = np.meshgrid(x1, x2, x3, indexing='ij')
    colormap = cm.get_cmap('RdBu')

    m_max = np.abs(moment).max()

    if m_max < 1e-10:
        m_max = 1
    x1s, x2s, x3s, colors = [[], [], []], [[], [], []], [[], [], []], [[], [], []]

    for i, m in np.ndenumerate(moment):
        x1s[i[-1]].append(i[0])
        x2s[i[-1]].append(i[1])
        x3s[i[-1]].append(i[2])
        colors[i[-1]].append(colormap(m / m_max / 2 + 0.5))

    for i, text in enumerate(("x", "y", "z")):
        plt.subplot(1, 3, i + 1)
        axes[i].scatter(x1s[i], x2s[i], x3s[i], color=colors[i])
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("y")
        axes[i].set_zlabel("z")
        plt.title(text + " component")


def get_fields(sol, _sol_rust, find_center, t, x1, x2, x3, current_moment, h_sym, t_sym, center,
               shift=0, magnetic=False, assuming_separability=True, delayed=True):
    c_mom = lambda a1, a2, a3: list(current_moment[:, a1, a2, a3])
    if assuming_separability:
        sol.set_moments(c_mom)
    kwargs = dict(compute_grid=False, shift=shift, delayed=delayed)

    if find_center:
        args = (x1 - center[0], x2 - center[1], x3 - center[2], t, h_sym, t_sym)
    else:
        args = (x1, x2, x3, t, h_sym, t_sym)
    if not magnetic:
        return sol.compute_e_field(*args, **kwargs)
    return sol.compute_b_field(*args, **kwargs)


def field_energy(x: np.ndarray, log=False, fit_on_derivative=False) -> np.ndarray:
    if fit_on_derivative:
        x = np.diff(x, axis=-1)
    if not log:
        return np.sum(x**2)
    e = x**2
    e[np.isclose(e, 0)] = np.nan
    return np.nansum(np.log10(e))


def inverse_problem(order, e_true, x1, x2, x3, t, _t_sym, current_moment_callable, dim_moment,
                    return_residual_error=False, **kwargs):

    print(f"{kwargs=}")

    if x1.size != x2.size or x2.size != x3.size or x1.size != x3.size:
        raise ValueError("Must provide a list of points, not a grid (i.e., :xi:, i=1,2,3 must have the same length)")

    tol = kwargs.pop("tol", 1e-1)
    n_points = kwargs.pop("n_points", 3)
    _error_tol = kwargs.pop("error_tol", 1e-1)
    _coeff_derivative = kwargs.pop("coeff_derivative", 0)
    verbose_every = kwargs.pop("verbose_every", 1)
    plot = kwargs.pop("plot", False)
    scale = kwargs.pop("scale", 1.)
    center_scale = kwargs.pop("center_scale", 1.)
    get_h_num = kwargs.pop("h_num", lambda _h, _t: _h)
    find_center = kwargs.pop("find_center", True)
    _max_global_tries = kwargs.pop("max_global_tries", 10)
    _compute_grid = kwargs.pop("compute_grid", True)
    _estimate = kwargs.pop("estimate", None)
    _p = kwargs.pop("p", 2)
    fit_on_magnetic_field = kwargs.pop("fit_on_magnetic_field", False)
    b_true = kwargs.pop("b_true", None)
    if fit_on_magnetic_field and b_true is None:
        raise ValueError("Must provide :b_true: when :fit_on_magnetic_field: is true")
    if fit_on_magnetic_field:
        scale /= 3e8
    rescale_at_points = kwargs.pop("rescale_at_points", False)
    fit_on_derivative = kwargs.pop("fit_on_derivative", False)
    return_x_opt = kwargs.pop("return_x_opt", False)
    random_start = kwargs.pop("random_start", True)
    delayed = kwargs.pop("delayed", True)
    find_center_ignore_axes = kwargs.pop(
        "find_center_ignore_axes",
        ()
    )
    minimize_kwargs = kwargs.pop("minimize_kwargs", {
        "method": "BFGS"
    })
    minimize_options = kwargs.pop("minimize_options", {
        'maxiter': 1_000_000_000,
        "disp": True,
        "gtol": tol
    })
    ignore_tail = kwargs.pop("ignore_tail", 0.)
    shift = kwargs.pop("shift", 0)
    seed = kwargs.pop("seed", 0)
    test_indices = set(kwargs.pop("test_indices", {}))
    train_indices = set(range(x1.size)) - test_indices
    train_indices, test_indices = list(sorted(train_indices)), list(sorted(test_indices))

    field_true = np.array(e_true if not fit_on_magnetic_field else b_true)
    tail = (t <= np.max(t) - ignore_tail)
    true_energy_train = field_energy(field_true[..., tail][:, train_indices, :], log=rescale_at_points,
                                     fit_on_derivative=fit_on_derivative)
    true_energy_test = field_energy(field_true[..., tail][:, test_indices, :], log=rescale_at_points,
                                    fit_on_derivative=fit_on_derivative)
    if kwargs:
        raise ValueError(f"Unknown keyword arguments: {kwargs}")

    _dt = np.max(np.diff(t))

    sol_python = pynoza.Solution(max_order=order,
                                 wave_speed=1, threshold=1e-28, )
    sol_python.recurse()
    # sol_rust = speenoza.Speenoza(order)

    n_center_coordinates = 3 - len(find_center_ignore_axes)
    center = np.zeros((n_center_coordinates, ))
    current_moment = np.zeros((dim_moment, ))
    h = np.zeros((n_points, ))

    _test = get_h_num(h, t)
    assuming_separability = isinstance(_test, np.ndarray)

    if find_center:
        def ravel_params(current_moments, h_, center_):
            return np.concatenate((np.ravel(current_moments), np.ravel(h_), np.ravel(center_)))

        def unravel_params(params):
            return params[:dim_moment], params[dim_moment:-n_center_coordinates], params[-n_center_coordinates:]
    else:
        def ravel_params(current_moments, h_, *_):
            return np.concatenate((np.ravel(current_moments), np.ravel(h_)))

        def unravel_params(params):
            return params[:dim_moment], params[dim_moment:], None

    x0 = ravel_params(current_moment, h, center)

    n_calls = 0
    field_opt = None

    _old_error = 100

    def get_error(x):
        nonlocal n_calls, field_opt, _old_error

        n_calls += 1

        current_moment_, h_, center_ = unravel_params(x)
        complete_center = [0., 0., 0.]
        if center_ is not None:
            ind_center_ = 0
            for i_coord, coordinate in enumerate(("x", "y", "z", )):
                if coordinate not in find_center_ignore_axes:
                    complete_center[i_coord] = center_[ind_center_] * center_scale
                    ind_center_ += 1

        h_ = get_h_num(h_, t)
        current_moment_ = current_moment_callable(current_moment_)
        field_opt = get_fields(sol_python, None, find_center, t, x1, x2, x3, current_moment_, h_, None,
                               complete_center, shift=shift, magnetic=fit_on_magnetic_field,
                               assuming_separability=assuming_separability, delayed=delayed)
        assert field_opt.shape == field_true.shape
        train_error = field_energy(field_true[..., tail][:, train_indices, :]
                                   - field_opt[..., tail][:, train_indices, :] * scale,
                                   log=rescale_at_points, fit_on_derivative=fit_on_derivative) / true_energy_train
        if true_energy_test > 1e-18:
            test_error = field_energy(field_true[..., tail][:, test_indices, :]
                                      - field_opt[..., tail][:, test_indices, :] * scale,
                                       log=rescale_at_points, fit_on_derivative=fit_on_derivative) / true_energy_test
        else:
            test_error = 1
        if n_calls % verbose_every == 0:
            if plot:
                plt.clf()
                max_true = np.max(np.abs(field_true))
                with pynoza.PlotAndWait(wait_for_enter_keypress=False):
                    for i, _color in enumerate(("r", "g", "b")):
                        plt.subplot(2, 3, i + 1)
                        plt.plot(t, field_true[i].reshape(-1, t.size).T, f"r--")
                        plt.plot(t, field_opt[i].reshape(-1, t.size).T*scale, f"k-")
                        plt.ylim((-1.1 * max_true, 1.1 * max_true))
                    if isinstance(h_, dict):
                        h_plot = np.array(list(h_.values()))
                    else:
                        h_plot = np.array(h_)
                    max_h = np.max(np.abs(h_plot))
                    plt.subplot(2, 3, 5)
                    if max_h > 0:
                        if len(h_plot.shape) == 3:
                            plt.plot(t, h_plot[:, 0, :].T, "r")
                            plt.plot(t, h_plot[:, 1, :].T, "g")
                            plt.plot(t, h_plot[:, 2, :].T, "b")
                        else:
                            plt.plot(t, h_plot)
                    if find_center:
                        plt.subplot(2, 3, 2)
                        plt.title(f"""center = ({complete_center[0]:+.03f}, """
                                  f"""{complete_center[1]:+.03f}, """
                                  f"""{complete_center[2]:+.03f})""")
            if plot:
                os.system("clear")
                end = "\r"
            else:
                end = "\n"
            print(f"{'#'*np.clip(int(train_error**.5*50), 0, 50)}{train_error**.5:.03f}, {n_calls=:},"
                  f" test: {test_error**.5:.03f}",
                  end=end)

        _old_error = train_error

        return train_error

    np.random.seed(seed)
    print(f"There are {x0.size} degrees of freedom.")

    n_calls = 0
    if random_start:
        x0 = np.random.random(x0.shape) * 2 - 1
        x0[-n_center_coordinates:] = np.zeros((n_center_coordinates, ))

    res = scipy.optimize.minimize(
        get_error, x0,
        options=minimize_options,
        **minimize_kwargs,
    )
    # res = scipy.optimize.basinhopping(
    #     get_error,
    #     x0,
    #     disp=True,
    #     niter=10,
    #     T=.1,
    #     accept_test=lambda f_new=None, **_: f_new < 0.3,
    # )
    current_moment, h, center = unravel_params(res.x)
    if find_center:
        center *= center_scale

    return_value = (current_moment_callable(current_moment), get_h_num(h, t), center, field_opt.squeeze() * scale, )

    if return_residual_error:
        return_value += (res.fun, )
    if return_x_opt:
        return_value += (res.x, )

    return return_value
