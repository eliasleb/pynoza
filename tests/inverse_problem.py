import pynoza
import numpy as np
import scipy.optimize
import scipy.interpolate
import matplotlib.pyplot as plt
import os
from matplotlib import cm
import speenoza


METHOD = "python"


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


def get_charge_moment(current_moment):
    charge_moment = np.zeros(current_moment.shape)
    for ind, _ in np.ndenumerate(charge_moment):
        i, a1, a2, a3 = ind
        a = (a1, a2, a3)
        for j in range(3):
            b = list(a)
            if i == j:
                if a[j] >= 2:
                    b[j] = a[j] - 2
                    charge_moment[i, a1, a2, a3] += a[j] * (a[j] - 1) \
                        * current_moment[j, b[0], b[1], b[2]]
            else:
                b[i] -= 1
                b[j] -= 1
                if a[j] >= 1 and a[i] >= 1:
                    charge_moment[i, a1, a2, a3] += a[j] * a[i] \
                        * current_moment[j, b[0], b[1], b[2]]
    return -charge_moment


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
    x1, x2, x3 = np.meshgrid(x1, x2, x3, indexing='ij')
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


def get_fields(sol_python, sol_rust, find_center, t, x1, x2, x3, current_moment, h_sym, t_sym, center, method="python"):
    if method == "python":
        pass
    c_mom = lambda a1, a2, a3: list(current_moment[:, a1, a2, a3])
    charge_moment = get_charge_moment(current_moment)
    r_mom = lambda a1, a2, a3: list(charge_moment[:, a1, a2, a3])
    sol_python.set_moments(c_mom, r_mom)
    if find_center:
        if method == "python":
            e_python = sol_python.compute_e_field(x1 - center[0],
                                                  x2 - center[1],
                                                  x3 - center[2],
                                                  t, h_sym, t_sym, compute_grid=False)
            return e_python
        else:
            e_rust = sol_rust.par_compute_e_field((x1 - center[0]).reshape(-1),
                                                  (x2 - center[1]).reshape(-1),
                                                  (x3 - center[2]).reshape(-1),
                                                  t.reshape(-1), h_sym.reshape(-1),
                                                  current_moment).swapaxes(1, 2)

            return e_rust
    else:
        if method == "python":
            return sol_python.compute_e_field(x1, x2, x3, t, h_sym, t_sym, compute_grid=False)
        else:
            return sol_rust.par_compute_e_field(x1.reshape(-1),
                                                x2.reshape(-1),
                                                x3.reshape(-1),
                                                t.reshape(-1), h_sym.reshape(-1),
                                                current_moment).swapaxes(1, 2)


def inverse_problem(order, e_true, x1, x2, x3, t, _t_sym, current_moment_callable, dim_moment, **kwargs):

    print(f"{kwargs=}")

    tol = kwargs.pop("tol", 1e-1)
    n_points = kwargs.pop("n_points", 3)
    error_tol = kwargs.pop("error_tol", 1e-1)
    coeff_derivative = kwargs.pop("coeff_derivative", 0)
    verbose_every = kwargs.pop("verbose_every", 1)
    plot = kwargs.pop("plot", False)
    scale = kwargs.pop("scale", 1)
    get_h_num = kwargs.pop("h_num", lambda h, t: h)
    find_center = kwargs.pop("find_center", True)
    max_global_tries = kwargs.pop("max_global_tries", 10)
    _compute_grid = kwargs.pop("compute_grid", True)
    estimate = kwargs.pop("estimate", None)
    p = kwargs.pop("p", 2)

    if kwargs:
        raise ValueError(f"Unknown keyword arguments: {kwargs}")

    dt = np.max(np.diff(t))

    #if METHOD == "python":
    sol_python = pynoza.Solution(max_order=order,
                          wave_speed=1, )
    sol_python.recurse()
   # else:
    sol_rust = speenoza.Speenoza(order)

    center = np.zeros((3, ))
    current_moment = np.zeros((dim_moment, ))
    h = np.zeros((n_points, ))

    if find_center:
        def ravel_params(current_moments, h_, center_):
            return np.concatenate((np.ravel(current_moments), np.ravel(h_), np.ravel(center_)))

        def unravel_params(params):
            return params[:dim_moment], params[dim_moment:-3], params[-3:]
    else:
        def ravel_params(current_moments, h_, *_):
            return np.concatenate((np.ravel(current_moments), np.ravel(h_)))

        def unravel_params(params):
            return params[:dim_moment], params[dim_moment:], None

    x0 = ravel_params(current_moment, h, center)

    n_calls = 0
    e_opt = None

    def get_error(x):
        nonlocal n_calls, e_opt

        n_calls += 1

        current_moment_, h_, center_ = unravel_params(x)
        h_ = get_h_num(h_, t)
        current_moment_ = current_moment_callable(current_moment_)
        e_opt = get_fields(sol_python, sol_rust, find_center, t, x1, x2, x3, current_moment_, h_, None, center_,
                           method=METHOD)

        total_energy = 0

        component_wise_error = []
        for c1, c2 in zip(e_true, e_opt):
            component_wise_error.append(np.sum(np.abs(c1 - c2 * scale)**p))
            total_energy += np.sum(np.abs(c1) ** p)
        error = np.sum(component_wise_error) / total_energy

        if coeff_derivative > 0:
            error += coeff_derivative*np.sum(np.diff(h)**2)/np.sum(h**2)*dt

        if n_calls % verbose_every == 0:
            if plot:
                plt.clf()
                max_true = np.max(np.abs(e_true))
                with pynoza.PlotAndWait(wait_for_enter_keypress=False):
                    for i in range(3):
                        plt.subplot(2, 3, i + 1)
                        plt.plot(t, e_true[i].reshape(-1, t.size).T, f"b--")
                        plt.plot(t, e_true[i].reshape(-1, t.size).T, f"b--")
                        plt.plot(t, e_true[i].reshape(-1, t.size).T, f"b--")

                        plt.plot(t, e_opt[i].reshape(-1, t.size).T*scale, f"k-")
                        plt.plot(t, e_opt[i].reshape(-1, t.size).T*scale, f"k-")
                        plt.plot(t, e_opt[i].reshape(-1, t.size).T*scale, f"k-")
                        plt.ylim((-1.1 * max_true, 1.1 * max_true))
                    max_h = np.max(np.abs(h_))
                    plt.subplot(2, 3, 5)
                    if max_h > 0:
                        plt.plot(t, h_ / max_h * max_true, "k-.")
                    if find_center:
                        plt.subplot(2, 3, 2)
                        plt.title(f"center = ({center_[0]:+.03f}, {center_[1]:+.03f} {center_[2]:+.03f})")

            os.system("clear")
            print(f"{'#'*np.clip(int(error*50), 0, 50)}{error:.3f}, {n_calls=}, {component_wise_error/total_energy=}",
                  end='\r')

        return error

    options = {'maxiter': 1e3,
               "disp": True,
               "gtol": tol
               }

    np.random.seed(0)
    print(f"There are {x0.size} degrees of freedom.")
    for i_try in range(max_global_tries):
        print("Try", i_try)
        if estimate is None:
            x0 = np.random.random(x0.shape) * 2 - 1
            x0[-3:] = np.array([0, 0, 0])
        else:
            x0 = ravel_params(*estimate)
        n_calls = 0
        res = scipy.optimize.minimize(get_error, x0,
                                      method="BFGS",
                                      options=options)

        if res.fun <= error_tol:
            break
    current_moment, h, center = unravel_params(res.x)

    return current_moment_callable(current_moment), get_h_num(h, t), center, e_opt.squeeze() * scale
