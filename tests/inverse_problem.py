import pynoza
import numpy as np
import scipy.optimize
import scipy.interpolate
import sympy
import matplotlib.pyplot as plt
import itertools
import sys
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


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
    b = [0, 0, 0]
    for ind, _ in np.ndenumerate(charge_moment):
        a1, a2, a3, i = ind
        a = (a1, a2, a3)
        for j in range(3):
            b = list(a)
            if i == j:
                if a[j] >= 2:
                    b[j] = a[j] - 2
                    charge_moment[a1, a2, a3, i] += a[j] * (a[j] - 1) \
                        * current_moment[b[0], b[1], b[2], j]
            else:
                b[i] -= 1
                b[j] -= 1
                if a[j] >= 1 and a[i] >= 1:
                    charge_moment[a1, a2, a3, i] += a[j] * a[i] \
                        * current_moment[b[0], b[1], b[2], j]
    return -charge_moment


def plot_moment(moment):
    fig = plt.figure()
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    axes = [ax1, ax2, ax3]

    order = moment.shape[0]
    x1 = np.arange(order)
    x2 = np.arange(order)
    x3 = np.arange(order)
    x1, x2, x3 = np.meshgrid(x1, x2, x3, indexing='ij')
    cmap = cm.get_cmap('RdBu')

    m_max = np.abs(moment).max()

    if m_max < 1e-10:
        m_max = 1
    x1s, x2s, x3s, colors = [[], [], []], [[], [], []], [[], [], []], [[], [], []]

    for i, m in np.ndenumerate(moment):
        x1s[i[-1]].append(i[0])
        x2s[i[-1]].append(i[1])
        x3s[i[-1]].append(i[2])
        colors[i[-1]].append(cmap(m / m_max / 2 + 0.5))

    for i, text in zip(range(3), ("x", "y", "z")):
        plt.subplot(1, 3, i + 1)
        axes[i].scatter(x1s[i], x2s[i], x3s[i], color=colors[i])
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("y")
        axes[i].set_zlabel("z")
        plt.title(text + " component")


def inverse_problem(order, e_true, x1, x2, x3, t, current_moment_callable, dim_moment, **kwargs):

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
    compute_grid = kwargs.pop("compute_grid", True)
    estimate = kwargs.pop("estimate", None)

    if kwargs:
        raise ValueError(f"Unknown keyword arguments: {kwargs}")

    if plot:
        plt.ion()
        plt.show()

    dt = np.max(np.diff(t))

    sol = pynoza.Solution(max_order=order,
                          wave_speed=1, )
    sol.recurse()

    def get_fields(current_moment, h_sym, t_sym, center):

        c_mom = lambda a1, a2, a3: list(current_moment[a1, a2, a3])
        charge_moment = get_charge_moment(current_moment)
        r_mom = lambda a1, a2, a3: list(charge_moment[a1, a2, a3])

        sol.set_moments(c_mom, r_mom)
        if find_center:
            return sol.compute_e_field(x1 - center[0],
                                       x2 - center[1],
                                       x3 - center[2],
                                       t,  h_sym, t_sym, compute_grid=compute_grid)
        else:
            return sol.compute_e_field(x1, x2, x3, t, h_sym, t_sym, compute_grid=compute_grid)

    center = np.zeros((3, ))
    current_moment = np.zeros((dim_moment, ))
    h = np.zeros((n_points, ))

    if find_center:
        def ravel_params(current_moments, h, center):
            return np.concatenate((np.ravel(current_moments), np.ravel(h), np.ravel(center)))

        def unravel_params(params):
            return params[:dim_moment], params[dim_moment:-3], params[-3:]
    else:
        def ravel_params(current_moments, h, *args):
            return np.concatenate((np.ravel(current_moments), np.ravel(h)))

        def unravel_params(params):
            return params[:dim_moment], params[dim_moment:], None

    x0 = ravel_params(current_moment, h, center)
    t_sym = sympy.Symbol("t", real=True)

    n_calls = 0
    old_error = 0
    e_opt = None

    def get_error(x):
        nonlocal n_calls, old_error, e_opt

        n_calls += 1

        current_moment, h, center = unravel_params(x)
        h = get_h_num(h, t)
        current_moment = current_moment_callable(current_moment)
        e_opt = get_fields(current_moment, h, None, center)

        error = 0
        normal = 0

        errors_comp = []

        for c1, c2 in zip(e_true, e_opt):
            errors_comp.append(np.sum(np.abs(c1 - c2 * scale)))
            normal += np.sum(np.abs(c1))
        error = np.sum(errors_comp) / normal

        if coeff_derivative > 0:
            error += coeff_derivative*np.sum(np.diff(h)**2)/np.sum(h**2)*dt

        error = np.clip(error, error_tol, 10)

        if n_calls % verbose_every == 0:
            if plot:
                plt.clf()
                #plt.subplot(2,1,1)
                max_true = np.max(np.abs(e_true))
                max_opt = np.max(np.abs(e_opt))
                colors = ["_r", "b", "g"]
                for i in range(3):
                    plt.plot(t, e_true[i].reshape(-1, t.size).T, f"{colors[i]}--")
                    plt.plot(t, e_true[i].reshape(-1, t.size).T, f"{colors[i]}--")
                    plt.plot(t, e_true[i].reshape(-1, t.size).T, f"{colors[i]}--")

                    plt.plot(t, e_opt[i].reshape(-1, t.size).T*scale, f"{colors[i]}-")
                    plt.plot(t, e_opt[i].reshape(-1, t.size).T*scale, f"{colors[i]}-")
                    plt.plot(t, e_opt[i].reshape(-1, t.size).T*scale, f"{colors[i]}-")

                #       print(np.max(np.abs(e_opt)), np.max(np.abs(current_moment)), np.max(np.abs(charge_moment)))
                max_h = np.max(np.abs(h))
                if max_h > 0:
                    plt.plot(t, h / max_h * max_true, "k-.")

                #plt.subplot(2, 1, 2)
                #directivity = np.sum(e_opt**2, axis=(0, 2))
                #max_directivity = directivity.max()
                #if max_directivity > 0:
                #    for xi, yi, zi, direc in zip(x1, x2, x3, directivity):
                #        if abs(zi):
                #            plt.plot(xi, yi, 'o', mfc=(direc/max_directivity, direc/max_directivity, direc/max_directivity), mec=(0,0,0))
                #plt.draw#()
                plt.pause(0.001)

            os.system("clear")
            print(f"{'#'*int(error*50)}{error:.3f}, {n_calls=}, {errors_comp/normal=}", end='\r')

        return error

    options = {'maxiter': 1e3,
               'disp': False,
               'seed': 0
               }

    np.random.seed(0)

    for i_try in range(max_global_tries):
        print("Try", i_try)
        if estimate is None:
            x0 = np.random.random(x0.shape) * 2 - 1
            x0[-3:] = np.array([0, 0, 0])
        else:
            x0 = ravel_params(*estimate)
        n_calls = 0

        res = scipy.optimize.minimize(get_error, x0,
                                      method=None,
                                      options=options, tol=tol, )

        if res.fun <= error_tol:
            break
    current_moment, h, center = unravel_params(res.x)

    return current_moment_callable(current_moment), get_h_num(h, t), center, e_opt.squeeze()
