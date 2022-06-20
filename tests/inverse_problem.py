import pynoza
import numpy as np
import scipy.optimize
import scipy.interpolate
import sympy
import matplotlib.pyplot as plt
import itertools
import sys
import os


def inverse_problem(order, e_true, x1, x2, x3, t, **kwargs):


    tol = kwargs.pop("tol", 1e-1)
    n_points = kwargs.pop("n_points", 3)
    error_tol = kwargs.pop("error_tol", 1e-1)
    coeff_derivative = kwargs.pop("coeff_derivative", 0)
    verbose_every = kwargs.pop("verbose_every", 1)
    plot = kwargs.pop("plot", False)
    scale = kwargs.pop("scale", 1)
    get_h_num = kwargs.pop("h_num", lambda h, t: h)
    find_center = kwargs.pop("find_center", True)

    if kwargs:
        raise ValueError(f"Unknown keyword arguments: {kwargs}")

    if plot:
        plt.ion()
        plt.show()

    dt = np.max(np.diff(t))

    sol = pynoza.Solution(max_order=order,
                          wave_speed=1, )
    sol.recurse()

    def get_fields(current_moment, charge_moment, h_sym, t_sym, center):
        c_mom = lambda a1, a2, a3: list(current_moment[a1, a2, a3])
        r_mom = lambda a1, a2, a3: list(charge_moment[a1, a2, a3])

        sol.set_moments(c_mom, r_mom)
        if find_center:
            return sol.compute_e_field(x1 - center[0],
                                       x2 - center[1],
                                       x3 - center[2],
                                       t,
                                       h_sym,
                                       t_sym)
        else:
            return sol.compute_e_field(x1, x2, x3, t, h_sym, t_sym)

    charge_moments = np.zeros((sol.max_order + 1, sol.max_order + 1, sol.max_order + 1, 3))
    current_moments = charge_moments.copy()
    Nmom = charge_moments.size
    shape_mom = charge_moments.shape

    h = np.zeros((n_points, ))
    Nh = h.size
    shape_h = h.shape
    center = np.zeros((3, ))


    def ravel_params(charge_moments, current_moments, h, center):
        return np.concatenate((np.ravel(charge_moments), np.ravel(current_moments), np.ravel(h), np.ravel(center)))

    def unravel_params(params):
        return params[:Nmom].reshape(shape_mom), \
               params[Nmom:Nmom + Nmom].reshape(shape_mom), \
               params[2 * Nmom:-3], params[-3:]

    x0 = ravel_params(charge_moments, current_moments, h, center)
    t_sym = sympy.Symbol("t", real=True)

    n_calls = 0

    def get_error(x):
        nonlocal n_calls
        n_calls += 1
        current_moment, charge_moment, h, center = unravel_params(x)
        h = get_h_num(h, t)
        e_opt = get_fields(current_moment, charge_moment, h, None, center)

        error = 0
        normal = 0

        for c1, c2 in zip(e_true, e_opt):
            error += np.sum((c1 - c2 * scale)**2)
            normal += np.sum(c1 ** 2)
        error = error / normal

        if coeff_derivative > 0:
            error += coeff_derivative*np.sum(np.diff(h)**2)/np.sum(h**2)*dt

        if n_calls % verbose_every == 0:
            if plot:
                plt.clf()
                #  e_opt = pynoza.set_extremities(e_opt, 0.1, dim=3)
                #       plt.plot(t, e_true[0][:, 0, 0, :].T, "--")
                max_true = np.max(np.abs(e_true))
                max_opt = np.max(np.abs(e_opt))
                colors = ["r", "b", "g"]
                for i in range(3):
                    plt.plot(t, e_true[i].reshape(-1, t.size).T, f"{colors[i]}--")
                    plt.plot(t, e_true[i].reshape(-1, t.size).T, f"{colors[i]}--")
                    plt.plot(t, e_true[i].reshape(-1, t.size).T/max_true, f"{colors[i]}--")

                    plt.plot(t, e_opt[i].reshape(-1, t.size).T*scale, f"{colors[i]}-")
                    plt.plot(t, e_opt[i].reshape(-1, t.size).T*scale, f"{colors[i]}-")
                    plt.plot(t, e_opt[i].reshape(-1, t.size).T*scale, f"{colors[i]}-")

                #       print(np.max(np.abs(e_opt)), np.max(np.abs(current_moment)), np.max(np.abs(charge_moment)))
                plt.plot(t, h/np.max(np.abs(h)) * max_true, "k-.")
                plt.draw()
                plt.pause(0.001)

            os.system("clear")
            print(f"{'#'*int(error*50)}{error:.3f}, {n_calls=}", end='\r')#f"\r{error=}, {n_calls=}")
        if error < error_tol:
            return 0
        else:
            return error

    options = {'maxiter': 1e3,
               'disp': True,
               }

    np.random.seed(0)
    x0 = np.random.random(x0.shape) * 2 - 1
    x0[-3:] = np.array([0, 0, 0])

    # res = scipy.optimize.basinhopping(get_error, x0, niter=0, stepwise_factor=0.1)
    res = scipy.optimize.minimize(get_error, x0, options=options, tol=tol, )
    current_moment, charge_moment, h, center = unravel_params(res.x)

    return current_moment, charge_moment, get_h_num(h, t), center
