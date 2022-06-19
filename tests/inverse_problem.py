import pynoza
import numpy as np
import scipy.optimize
import scipy.interpolate
import sympy
import matplotlib.pyplot as plt
import itertools
import sys


def inverse_problem(order, e_true, x1, x2, x3, t, **kwargs):

    tol = kwargs.pop("tol", 1e-1)
    n_points = kwargs.pop("n_points", 3)
    error_tol = kwargs.pop("error_tol", 1e-1)
    coeff_derivative = kwargs.pop("coeff_derivative", 0)

    dt = np.max(np.diff(t))

    sol = pynoza.Solution(max_order=order,
                          wave_speed=1, )
    sol.recurse()

    def get_fields(current_moment, charge_moment, h_sym, t_sym):
        c_mom = lambda a1, a2, a3: list(current_moment[a1, a2, a3])
        r_mom = lambda a1, a2, a3: list(charge_moment[a1, a2, a3])

        sol.set_moments(c_mom, r_mom)

        return sol.compute_e_field(x1,
                                   x2,
                                   x3,
                                   t,
                                   h_sym,
                                   t_sym)

    charge_moments = np.zeros((sol.max_order + 1, sol.max_order + 1, sol.max_order + 1, 3))
    current_moments = charge_moments.copy()
    Nmom = charge_moments.size
    shape_mom = charge_moments.shape

    h = np.zeros((n_points, ))
    Nh = h.size
    shape_h = h.shape

    def get_h_num(h):
        return scipy.interpolate.interp1d(np.linspace(t.min(), t.max(), h.size), h,
                                          kind="quadratic")(t)

    def ravel_params(charge_moment, current_moments, h):
        return np.concatenate((np.ravel(charge_moment), np.ravel(current_moments), np.ravel(h)))

    def unravel_params(params):
        return params[:Nmom].reshape(shape_mom), \
               params[Nmom:Nmom + Nmom].reshape(shape_mom), \
               params[2 * Nmom:]

    x0 = ravel_params(charge_moments, current_moments, h)
    t_sym = sympy.Symbol("t", real=True)

    n_calls = 0

    def get_error(x):
        nonlocal n_calls
        n_calls += 1
        current_moment, charge_moment, h = unravel_params(x)
        h = get_h_num(h)
        e_opt = get_fields(current_moment, charge_moment, h, t_sym)
        #  e_opt = pynoza.set_extremities(e_opt, 0.1, dim=3)

        error = 0
        normal = 0
        print(e_true[0].shape, e_opt[0].shape)
        for c1, c2 in zip(e_true, e_opt):
            error += np.sum((c1 - c2 / 377) ** 2)
            normal += np.sum(c1 ** 2)
        error = error / normal
        if coeff_derivative > 0:
            error += coeff_derivative*np.sum(np.diff(h)**2)/np.sum(h**2)*dt

        if n_calls % 1 == 0:
            sys.stdout.write(f"\r{error=}, {n_calls=}")

        if error < error_tol:
            return 0
        else:
            return error

    options = {'maxiter': 1e3,
               'disp': True,
               }

    np.random.seed(0)
    x0 = np.random.random(x0.shape)

    # res = scipy.optimize.basinhopping(get_error, x0, niter=0, stepwise_factor=0.1)
    res = scipy.optimize.minimize(get_error, x0, options=options, tol=tol)
    current_moment, charge_moment, h = unravel_params(res.x)

    return current_moment, charge_moment, get_h_num(h)
