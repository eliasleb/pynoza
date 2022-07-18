#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 11:16:58 2021

@author: Elias Le Boudec, elias.leboudec@epfl.ch
"""
import numpy as np
from numpy import zeros
from numpy import ndarray
from numpy import sum as np_sum
import sys
import sympy
import scipy.interpolate
import numbers
import cython


class Interpolator(scipy.interpolate.interp1d):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__doc__ = scipy.interpolate.interp1d.__doc__

    def __call__(self, *args, **kwargs) -> ndarray:
        x: ndarray = args[0].copy()
        idx: ndarray = (x < self.x.min()) | (x > self.x.max())

        x[x < self.x.min()] = self.x.min()
        x[x > self.x.max()] = self.x.max()

        y: ndarray = super().__call__(*((x, ) + args[1:]), **kwargs)
        y[idx] = 0
        return y


X1: int = 0
X2: int = 1
X3: int = 2
H: int = 3
R: int = 4


@cython.cclass
class Solution:
    """A class to compute solutions of Maxwell's equations, based on"""
    """time-domain multipole moments."""
    max_order: int
    c: float
    _shape: tuple
    _aux_func: dict
    mu: float
    thresh: float
    ran_recurse: bool
    ran_set_moments: bool
    current_moment: cython.ccall
    charge_moment: cython.ccall
    verbose: bool
    delayed: bool
    compute_grid: bool
    e_field: ndarray
    _r: ndarray
    y: ndarray
    dy: ndarray
    e_field_text: str

    def __init__(self,
                 max_order: int = 0,
                 wave_speed: int = 1) -> None:
        """Initialize the solution class.
        
        Keyword arguments:
        max_order -- the maximum order to which multipole moments will be computed (default 0)
        wave_speed -- Wave speed `c` used to compute the retarded time t-r/c, in natural units (default 1)
        """""
        if not isinstance(max_order, int):
            raise ValueError(":max_order: must be of integer type")
        if not isinstance(wave_speed, numbers.Number):
            raise ValueError(":wave_speed: must be a number")

        self.max_order = max_order
        self.c = wave_speed
        self._shape = (max_order + 1, max_order + 1, max_order + 1)
        self._aux_func = dict()
        for ind, _ in np.ndenumerate(zeros(self._shape)):
            self._aux_func[tuple(ind)] = dict()
        self._aux_func[(0, 0, 0)] = {(0, 0, 0, 0, 1): 1.}
        self.mu: float = 4 * np.pi * 1e-7
        self.thresh: float = 1e-14

        self.ran_recurse = False
        self.ran_set_moments = False
        self.current_moment = None
        self.charge_moment = None
        self.verbose = False
        self.delayed = True
        self.compute_grid = True

        self.e_field: ndarray = np.array([])

        self.e_field_text = ""
        self._r = np.array([0, ])
        self.y = np.array([0, ])
        self.dy = np.array([0, ])

    def _increase_order(self, known_index, index) -> None:
        """Private method to compute the auxiliary function.
        
        Positional arguments:
        known_index -- multi-index at which the auxiliary function has already been computed
        index -- multi-index to compute the auxiliary function"""
        known_dim: int = np.where(np.array(known_index) - np.array(index))[0][0]
        for signature in self._aux_func[known_index]:
            coefficient: float = self._aux_func[known_index][signature]
            # We need to apply the recursion formula to this term.
            # This adds three new terms; two for the space-derivative, and one for the time-derivative
            exponent_x_i: int = signature[known_dim]
            if exponent_x_i > 0:
                identity_first_term = list(signature)
                identity_first_term[known_dim] -= 1  # differentiate
                try:
                    self._aux_func[index][tuple(identity_first_term)] \
                        += coefficient * exponent_x_i
                except KeyError:
                    self._aux_func[index][tuple(identity_first_term)] \
                        = coefficient * exponent_x_i
            exponent_r: int = signature[R]
            identity_second_term = list(signature)
            identity_second_term[known_dim] += 1  # numerator
            identity_second_term[R] += 2  # denominator
            try:
                self._aux_func[index][tuple(identity_second_term)] -= \
                    coefficient * exponent_r
            except KeyError:
                self._aux_func[index][tuple(identity_second_term)] = \
                    -coefficient * exponent_r
            # Time-derivative term
            identity_third_term = list(signature)
            identity_third_term[known_dim] += 1
            identity_third_term[H] += 1  # time-derivative
            identity_third_term[R] += 1  # denominator
            try:
                self._aux_func[index][tuple(identity_third_term)] -= \
                    coefficient / self.c
            except KeyError:
                self._aux_func[index][tuple(identity_third_term)] = \
                    -coefficient / self.c

    def recurse(self, verbose: bool = False) -> None:
        """Compute the auxiliary function up to the max order.
        
        Keyword arguments:
        verbose -- whether to print the computed multi-index (default True)"""
        self.ran_recurse = True
        for order in range(1, self.max_order + 1):
            for ind, _ in np.ndenumerate(zeros(self._shape)):
                ind: ndarray = np.array(ind)
                if np_sum(ind) == order:
                    known_ind: np.array = ind.copy()
                    known_ind[np.where(ind > 0)[0][0]] -= 1
                    if verbose:
                        sys.stdout.write("\rComputing order {}...".format(order))
                    self._increase_order(tuple(known_ind), tuple(ind))
        if verbose:
            print("Done.")

    @cython.ccall
    def _evaluate(self,
                  ind,
                  x1: ndarray,
                  x2: ndarray,
                  x3: ndarray,
                  r: ndarray,
                  hs):
        """Evaluate the auxiliary function.
        
        Positional arguments:
        ind -- multi-index of the auxiliary function
        X1 -- evaluated first coordinate (aka x)
        X2 -- evaluated second coordinate (aka y)
        X3 -- evaluated third coordinate (aka z)
        R -- equal to X1**2+X2**2+X3**2. Passed to avoid computing it repeatedly.
        Hs -- dictionary of the derivatives of the time-dependent excitation function.
              Must be in the form {order:derivative of order} for order=-1..max_order+2
        """
        self.y = zeros(x1.shape)
        for signature in self._aux_func[ind]:
            self.dy = hs[signature[H]] * self._aux_func[ind][signature]
            if signature[X1] > 0:
                self.dy = self.dy * x1 ** signature[X1]
            if signature[X2] > 0:
                self.dy = self.dy * x2 ** signature[X2]
            if signature[X3] > 0:
                self.dy = self.dy * x3 ** signature[X3]
            if signature[R] > 0:
                self.dy = self.dy / r ** signature[R]
            self.y = self.y + self.dy
        return self.y

    def _evaluate_txt(self,
                      ind,
                      h: str) -> str:
        """Evaluate the auxiliary function as a symbolic expression

            param ind: multi-index to evaluate at
            param h: name of the function
            return: a string describing the auxiliary function
            """
        y: str = ""
        for signature in self._aux_func[ind]:
            dy = ""
            dy += f"{self._aux_func[ind][signature]:.2e}*{h}^({signature[H]})(t-r/{self.c:.1f})"
            if signature[X1] > 0:
                dy += f"x1^{signature[X1]}"
            if signature[X2] > 0:
                dy += f"x2^{signature[X2]}"
            if signature[X3] > 0:
                dy += f"x3^{signature[X3]}"
            if signature[R] > 0:
                dy += f"/r^{signature[R]}"
            y += dy
        return y

    def set_moments(self,
                    current_moment=lambda a1, a2, a3: [0, 0, 0],
                    charge_moment=lambda a1, a2, a3: [0, 0, 0],) -> None:
        """Set the current and charge moment functions.
        
        Keyword arguments:
        current_moment -- a callable returning the current moment for a given multi-index a1,a2,a3
        charge_moment -- a callable returning the charge moment for a given multi-index a1,a2,a3"""

        if not callable(current_moment) or not callable(charge_moment):
            raise ValueError(":current_moment: and :charge_moment: must be callable")
        if not isinstance(current_moment(0, 0, 0), list) \
                or not isinstance(charge_moment(0, 0, 0), list):
            raise ValueError(":current_moment: and :charge_moment: callables must return a list of numbers")

        self.ran_set_moments = True
        self.current_moment = np.zeros((3, self.max_order + 1, self.max_order + 1, self.max_order + 1))
        self.charge_moment = self.current_moment.copy()

        for ind, _ in np.ndenumerate(zeros(self._shape)):
            self.current_moment[:, ind[0], ind[1], ind[2]] = current_moment(*ind)
            self.charge_moment[:, ind[0], ind[1], ind[2]] = charge_moment(*ind)

    @cython.ccall
    def compute_e_field(self,
                        x1: ndarray,
                        x2: ndarray,
                        x3: ndarray,
                        t: ndarray,
                        h_sym,
                        t_sym,
                        verbose=False,
                        delayed=True,
                        compute_grid=True,
                        compute_txt=False):
        """Compute the electric field from the moments.
        
        Positional arguments:
        x1,x2,x3 -- arrays of the spatial coordinates to evaluate the e_field-field at (aka x,y,z)
        t -- array of the time coordinates to evaluate the e_field-field at
        h_sym -- symbolic time-dependent (t_sym) function describing the shape of the current
                 *or* a dictionary of numpy arrays of the same shape as t, each array containing the
                  values of the nth order derivative of the time-dependent function. The keys of the
                  dictionary must be the integers in the range -1..max_order+2.
        t_sym -- symbolic variable representing time, used in h_sym
        
        Keyword arguments:
        verbose -- whether to display the progress (default False)
        delayed -- whether to evaluate the field at the retarded time t-r/c (default True)"""

        if not self.ran_recurse or not self.ran_set_moments:
            raise RuntimeError("You must first run the `recurse' and `set_moments' methods.")

        if isinstance(h_sym, dict):
            if not set(h_sym.keys()).issuperset(set(range(-1, self.max_order + 3))):
                raise ValueError("When h_sym is a dictionary, the keys must contain"
                                 " the indices -1..max_order + 2")

        self.verbose = verbose
        self.delayed = delayed
        self.compute_grid = compute_grid

        compute_txt = compute_txt

        if self.verbose:
            np.seterr(divide="raise",
                      over="raise",
                      under="warn",
                      invalid="raise")
        else:
            np.seterr(divide="raise",
                      over="raise",
                      under="ignore",
                      invalid="raise")

        self.e_field_text = ""

        if self.compute_grid:
            x1 = x1.reshape((x1.size, 1, 1, 1))
            x2 = x2.reshape((1, x2.size, 1, 1))
            x3 = x3.reshape((1, 1, x3.size, 1))
            t = t.reshape((1, 1, 1, t.size))
            moment_shape = (3, 1, 1, 1, 1)
            self.e_field = zeros((3, x1.size, x2.size, x3.size, t.size))
        else:
            x1 = x1.reshape((x1.size, 1))
            x2 = x2.reshape((x2.size, 1))
            x3 = x3.reshape((x3.size, 1))
            t = t.reshape((1, t.size))
            moment_shape = (3, 1, 1)
            self.e_field = zeros((3, x1.size, t.size))

        self._r = np.sqrt(x1 ** 2 + x2 ** 2 + x3 ** 2)

        if isinstance(h_sym, ndarray):
            hs_derivative, hs_integral = self._handle_h_array(h_sym, t)
        else:
            hs_derivative, hs_integral = self._handle_h_symbolic(h_sym, t_sym, t)

        for ind, _ in np.ndenumerate(zeros(self._shape)):
            ind = np.array(ind)
            if np_sum(ind) <= self.max_order:
                a1, a2, a3 = ind
                if self.verbose:
                    sys.stdout.write("\rComputing index {}...".format(ind))
                charge_moment = -self.mu * self.c ** 2 * self.charge_moment[:, a1, a2, a3].reshape(moment_shape)
                current_moment = -self.mu * self.current_moment[:, a1, a2, a3].reshape(moment_shape)
                if np.any(charge_moment) > self.thresh:
                    self.e_field += self._single_term_multipole(ind,
                                                                charge_moment,
                                                                hs_integral,
                                                                x1, x2, x3, self._r)
                    if compute_txt:
                        self.e_field_text += self._single_term_multipole_txt(ind,
                                                                             charge_moment,
                                                                             "int_h")
                if np.any(current_moment) > self.thresh:
                    self.e_field += self._single_term_multipole(ind,
                                                                current_moment,
                                                                hs_derivative,
                                                                x1, x2, x3, self._r)
                    if compute_txt:
                        self.e_field_text += self._single_term_multipole_txt(ind,
                                                                             current_moment,
                                                                             "dh_dt")

        if self.verbose:
            print("Done.")

        return self.e_field

    def _handle_h_symbolic(self, h_sym, t_sym, t):
        h_0_sym = h_sym.integrate(t_sym)
        hs_sym = {-1: h_0_sym, 0: h_sym}
        for order in range(1, self.max_order + 3):
            if self.verbose:
                sys.stdout.write("\rComputing derivative of order {}...".format(order))
            hs_sym[order] = sympy.diff(hs_sym[order - 1]).simplify()
        if self.verbose:
            print("Done.")
        hs = dict()
        for order in hs_sym:
            if self.delayed:
                hs[order] = sympy.lambdify(t_sym, hs_sym[order])(t - self._r / self.c)
            else:
                hs[order] = sympy.lambdify(t_sym, hs_sym[order])(t)

        return self._repack_hs(hs)

    def _repack_hs(self, hs):
        hs_integral = list()
        hs_derivative = list()
        for order in range(-1, self.max_order + 3):
            if order > 0:
                if order <= self.max_order:
                    hs_derivative.append(hs[order])
                    hs_integral.append(hs[order])
                else:
                    hs_derivative.append(0)
                    hs_integral.append(0)
            else:
                hs_integral.append(hs[order])

        return hs_derivative, hs_integral

    def _handle_h_array(self, h, t):

        dt = np.max(np.diff(t))

        def integrate_array(x):
            return np.cumsum(x) * dt

        def derivative(x):
            return np.gradient(x, dt)

        hs = {-1: integrate_array(h), 0: h}

        for i in range(1, self.max_order + 3):
            hs[i] = derivative(hs[i - 1])

        h_sym_callable = dict()
        for order in hs:
            if self.delayed:
                h_sym_callable[order] = Interpolator(t.squeeze(), hs[order])(t - self._r / self.c)
            else:
                h_sym_callable[order] = Interpolator(t.squeeze(), hs[order])(t)

        return self._repack_hs(h_sym_callable)

    def _single_term_multipole(self,
                               ind: ndarray,
                               moment: ndarray,
                               hs: ndarray,
                               *args):
        return (-1) ** np_sum(ind) / fact(ind) \
               * moment * self._evaluate(tuple(ind), *args, hs) / 4 / np.pi

    def _single_term_multipole_txt(self,
                                   ind: ndarray,
                                   moment: ndarray,
                                   hs: str):
        return f"""  {(-1) ** np_sum(ind):+d}/{fact(ind)}"""\
               f"""*{list(map('{:.2e}%'.format, moment.flatten()))}*{self._evaluate_txt(tuple(ind), hs)}/(4pi)\n"""

    def __repr__(self) -> str:
        return f"Solution: max_order={self.max_order}, c={self.c}, ran_recurse={self.ran_recurse}"

    def get_e_field_text(self):
        return self.e_field_text


def fact(a) -> numbers.Number:
    res: numbers.Number = 1
    for i in a:
        res *= np.math.factorial(i)
    return res


def set_extremities(x, ratio, dim=0, val=0):

    portion = np.linspace(0, 1, x.shape[dim])
    start = np.where(portion > ratio/2)[0][0]
    stop = np.where(portion > 1 - ratio/2)[0][0]
    s_start = {dim: slice(0, start)}
    s_stop = {dim: slice(stop, None)}
    ix_start = [s_start.get(dim, slice(None)) for dim in range(x.ndim)]
    ix_stop = [s_stop.get(dim, slice(None)) for dim in range(x.ndim)]

    x[tuple(ix_start)] = val
    x[tuple(ix_stop)] = val

    return x


if __name__ == "__main__":
    pass
