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
from scipy.interpolate import interp1d
from scipy.special import factorial
import numbers
import cython

import pynoza.helpers


class Interpolator(scipy.interpolate.interp1d):
    """
    A 1D interpolator that extrapolates with zeros.

    :param args: See scipy.interpolate.interp1d
    :param kwargs: scipy.interpolate.interp1d
    """
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
    """
    A class to compute solutions of Maxwell's equations, based on time-domain multipole moments.
    """
    max_order: int
    c: float
    _shape: tuple
    _aux_func: dict
    mu: float
    thresh: float
    ran_recurse: bool
    ran_set_moments: bool
    verbose: bool
    delayed: bool
    compute_grid: bool
    e_field: ndarray
    b_field: ndarray
    _r: ndarray
    y: ndarray
    dy: ndarray
    e_field_text: str
    b_field_text: str
    _causal: bool
    current_moment: ndarray
    magnetic_moment: ndarray
    charge_moment: ndarray
    _rho_to_j_mapping: dict

    def __init__(
            self, max_order: int = 0, wave_speed: float = 1., causal=True, threshold=1e-14
    ) -> None:
        """
        Initialize the solution class.
        
        :param max_order: The maximum order to which multipole moments will be computed (default 0)
        :param wave_speed: The wave speed `c` used to compute the retarded time t-r/c, in natural units (default 1)
        :param threshold: Minimum moment absolute value to be considered for computations
        """""
        if not isinstance(max_order, int):
            raise TypeError(":max_order: must be of integer type")
        if not isinstance(wave_speed, numbers.Number):
            raise TypeError(":wave_speed: must be a number")

        self.max_order = max_order
        self.c = wave_speed
        self._shape = (max_order + 1, max_order + 1, max_order + 1)
        self._aux_func = dict()
        for ind, _ in np.ndenumerate(zeros(self._shape)):
            self._aux_func[tuple(ind)] = dict()
        self._aux_func[(0, 0, 0)] = {(0, 0, 0, 0, 1): 1.}
        self.mu: float = 4 * np.pi * 1e-7
        self.thresh: float = threshold

        self.ran_recurse = False
        self.ran_set_moments = False
        self.current_moment = None
        self.magnetic_moment = None
        self.charge_moment = None
        self.verbose = False
        self.delayed = True
        self.compute_grid = True
        self._rho_to_j_mapping = {}

        self.e_field: ndarray = np.array([])
        self.b_field: ndarray = np.array([])

        self.e_field_text = ""
        self.b_field_text = ""
        self._r = np.array([0, ])
        self.y = np.array([0, ])
        self.dy = np.array([0, ])
        self._causal = causal

    def _increase_order(self, known_index, index) -> None:
        """
        Method to compute the auxiliary function.

        :param known_index: multi-index at which the auxiliary function has already been computed
        :param index: multi-index to compute the auxiliary function"""
        known_dim: np.int64 = np.where(np.array(known_index) - np.array(index))[0][0]
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
                    coefficient / self.c * (1 if self._causal else -1)
            except KeyError:
                self._aux_func[index][tuple(identity_third_term)] = \
                    -coefficient / self.c * (1 if self._causal else -1)

    def get_mu(self):
        return self.mu

    def get_magnetic_moment(self):
        return self.magnetic_moment

    def recurse(self, verbose: bool = False) -> None:
        """Compute the auxiliary function up to the max order.
        
        :param verbose: whether to print the computed multi-index (default False)"""
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
                  hs: list | dict):
        """
        Evaluate the auxiliary function.

        :param ind: multi-index of the auxiliary function
        :param x1: evaluated first coordinate (aka x)
        :param x2: evaluated second coordinate (aka y)
        :param x3: evaluated third coordinate (aka z)
        :param r: equal to X1**2+X2**2+X3**2. Passed to avoid computing it repeatedly.
        :param hs: derivatives of the time-dependent excitation function. Must be in the form
        {order: derivative of order} for order=-1..max_order+2
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
        """
        Evaluate the auxiliary function as a symbolic expression

        :param ind: multi-index to evaluate at
        :param h: name of the function
        :return: a string describing the auxiliary function
        """
        y: str = "("
        for signature in self._aux_func[ind]:
            dy = "+("
            dy += f"{self._aux_func[ind][signature]:.2e}*{h}^({signature[H]})(t-r/{self.c:.1f})"
            if signature[X1] > 0:
                dy += f"*x1^{signature[X1]}"
            if signature[X2] > 0:
                dy += f"*x2^{signature[X2]}"
            if signature[X3] > 0:
                dy += f"*x3^{signature[X3]}"
            if signature[R] > 0:
                dy += f"/r^{signature[R]}"
            y += dy + ")"
        return y + ")"

    def set_moments(self,
                    current_moment=lambda a1, a2, a3: [0, 0, 0],
                    charge_moment=None) -> None:
        """
        Set the current and charge moment functions.

        :param current_moment: a callable returning the current moment for a given multi-index a1, a2, a3
        :param charge_moment: a callable returning the charge moment for a given multi-index a1, a2, a3 (optional)

        If the parameter :charge_moment: is not given, it is automatically computed according to
        :pynoza.helpers.get_charge_moment:.
        """

        if not callable(current_moment):
            raise TypeError(":current_moment: must be callable")
        if not isinstance(current_moment(0, 0, 0), list):
            raise ValueError(":current_moment: callable must return a list of numbers")
        if charge_moment is not None:
            if not callable(charge_moment):
                raise TypeError(":charge_moment: must be callable")
            if not isinstance(charge_moment(0, 0, 0), list):
                raise ValueError(":charge_moment: callable must return a list of numbers")

        self.ran_set_moments = True
        self.current_moment = np.zeros((3, self.max_order + 1, self.max_order + 1, self.max_order + 1))
        self.charge_moment = self.current_moment.copy()
        self.magnetic_moment = self.current_moment.copy()

        for ind, _ in np.ndenumerate(zeros(self._shape)):
            self.current_moment[:, ind[0], ind[1], ind[2]] = current_moment(*ind)
            if charge_moment is not None:
                self.charge_moment[:, ind[0], ind[1], ind[2]] = charge_moment(*ind)
        if charge_moment is None:
            self.charge_moment, _rho_to_j_mapping = pynoza.helpers.get_charge_moment(
                self.current_moment, return_mapping=True)
            self._rho_to_j_mapping = dict()
            for k, v in _rho_to_j_mapping.items():
                assert len(v) == 1
                self._rho_to_j_mapping[k] = v.pop()
        self.magnetic_moment = pynoza.helpers.get_magnetic_moment(self.current_moment)

    @cython.ccall
    def _prepare_arguments(
            self,
            x1: ndarray,
            x2: ndarray,
            x3: ndarray,
            t: ndarray,
            h_sym,
            t_sym,
            verbose,
            delayed,
            compute_grid,
            shift,
            magnetic
    ):
        if not self.ran_recurse or not self.ran_set_moments:
            raise RuntimeError("You must first run the `recurse' and `set_moments' methods.")

        if isinstance(h_sym, dict):
            if isinstance(list(h_sym.keys())[0], int):
                if not set(h_sym.keys()).issuperset(set(range(-1, self.max_order + 3))):
                    raise ValueError("When h_sym is a dictionary with integer keys, the keys must contain"
                                     " the indices -1..max_order + 2")
            elif isinstance(list(h_sym.keys())[0], tuple):
                if len(list(h_sym.keys())[0]) != 3:
                    raise ValueError("When h_sym is a dictionary with tuple keys, the keys must have length 3")
                # if len(list(h_sym.values())[0]) != 3:
                #     raise ValueError("When h_sym is a dictionary with tuple keys, the values must have length 3")
            else:
                raise ValueError("When h_sym is a dictionary, the keys must either be integers (derivative orders)"
                                 " or tuples (dim, ax, ay, az)")

        self.verbose = verbose
        self.delayed = delayed
        self.compute_grid = compute_grid

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

        if magnetic:
            self.b_field_text = ""
        else:
            self.e_field_text = ""

        if self.compute_grid:
            x1 = x1.reshape((x1.size, 1, 1, 1))
            x2 = x2.reshape((1, x2.size, 1, 1))
            x3 = x3.reshape((1, 1, x3.size, 1))
            t = t.reshape((1, 1, 1, t.size))
            moment_shape = (3, 1, 1, 1, 1)
            _array = zeros((3, x1.size, x2.size, x3.size, t.size))
        else:
            x1 = x1.reshape((x1.size, 1))
            x2 = x2.reshape((x2.size, 1))
            x3 = x3.reshape((x3.size, 1))
            t = t.reshape((1, t.size))
            moment_shape = (3, 1, 1)
            _array = zeros((3, x1.size, t.size))

        if magnetic:
            self.b_field = _array
        else:
            self.e_field = _array

        self._r = np.sqrt(x1 ** 2 + x2 ** 2 + x3 ** 2)
        if isinstance(h_sym, ndarray):
            hs_0, hs_derivative, hs_integral = self._handle_h_array(h_sym, t, shift=shift)
            hs_0, hs_derivative, hs_integral = {None: hs_0}, {None: hs_derivative}, {None: hs_integral}
        elif isinstance(h_sym, dict):
            hs_0, hs_derivative, hs_integral = self._handle_h_dict(h_sym, t, shift=shift)
            hs_0[None] = None
            hs_derivative[None] = None
            hs_integral[None] = None
        else:
            hs_0, hs_derivative, hs_integral = self._handle_h_symbolic(h_sym, t_sym, t)
            hs_0, hs_derivative, hs_integral = {None: hs_0}, {None: hs_derivative}, {None: hs_integral}
        if magnetic:
            return x1, x2, x3, t, moment_shape, hs_0
        return x1, x2, x3, t, moment_shape, hs_derivative, hs_integral

    @cython.ccall
    def compute_e_field(
        self,
        x1: ndarray,
        x2: ndarray,
        x3: ndarray,
        t: ndarray,
        h_sym,
        t_sym,
        verbose=False,
        delayed=True,
        compute_grid=True,
        compute_txt=False,
        shift=0,
    ):
        """
        Compute the electric field from the moments. The method `recurse()` and `set_moments(...)` must be run
        beforehand.

        :return: the electric field as a 5-dimensional array. If :compute_grid: is True, the dimensions correspond to
         (dimension, x1, x2, x3, t), otherwise, (dimension, x, t).
        :param x1: array of the spatial coordinates to evaluate the e_field-field at (aka x)
        :param x2: array of the spatial coordinates to evaluate the e_field-field at (aka y)
        :param x3: array of the spatial coordinates to evaluate the e_field-field at (aka z)
        :param t: array of the time coordinates to evaluate the e_field-field at
        :param h_sym: time-dependent excitation
        :param t_sym: symbolic variable representing time, used in h_sym
        :param verbose: whether to display the progress (default False)
        :param delayed: whether to evaluate the field at the retarded time t-r/c (default True)
        :param compute_grid: whether to compute all combinations of coordinates
        :param compute_txt: whether to compute a text representation of the solution or not
        :param shift: experimental

        :raises RuntimeError: when any of `recurse` or `set_moments` have not been run.
        :raises ValueError: when `h_sym` does not look like what is described below

        :rtype: a np.ndarray with the electric field, whose shape is `(3, x1.size, t.size)`

        The time-dependent excitation `h_sym` can either be a symbolic time-dependent (t_sym) function describing the
        shape of the current *or* a dictionary of numpy arrays of the same shape as t, each array containing the values
        of the nth order derivative of the time-dependent function. The keys of the dictionary must be the integers in
        the range -1..max_order+2.
        """
        x1, x2, x3, t, moment_shape, hs_derivative, hs_integral = self._prepare_arguments(
            x1, x2, x3, t, h_sym, t_sym,
            verbose=verbose,
            delayed=delayed,
            compute_grid=compute_grid,
            shift=shift,
            magnetic=False
        )

        for ind, _ in np.ndenumerate(zeros(self._shape)):
            ind = np.array(ind)
            if np_sum(ind) <= self.max_order:
                a1, a2, a3 = ind
                if self.verbose:
                    sys.stdout.write("\rComputing index {}...".format(ind))
                charge_moment = -self.mu * self.c ** 2 * self.charge_moment[:, a1, a2, a3].reshape(moment_shape)
                current_moment = -self.mu * self.current_moment[:, a1, a2, a3].reshape(moment_shape)
                if np.any(np.abs(charge_moment) > self.thresh):
                    hs_charge = None
                    for dim, charge_moment_i in enumerate(charge_moment):
                        if np.abs(charge_moment_i) < self.thresh:
                            continue
                        current_index = self._rho_to_j_mapping.get((dim, ) + tuple(ind), None)
                        hi = hs_integral.get(
                            current_index[1:], hs_integral[None]
                        )
                        if hs_charge is None:
                            shape = (3, ) + hi[0].shape
                            hs_charge = [np.zeros(shape) for _ in range(len(hi))]
                        for ind_hij, hij in enumerate(hi):
                            hs_charge[ind_hij][dim, ...] = hij
                    self.e_field += self._single_term_multipole(ind,
                                                                charge_moment,
                                                                hs_charge,
                                                                x1, x2, x3, self._r)
                    if compute_txt:
                        self.e_field_text += self._single_term_multipole_txt(
                            ind, charge_moment, "int_h")
                if np.any(np.abs(current_moment) > self.thresh):
                    self.e_field += self._single_term_multipole(
                        ind, current_moment, hs_derivative.get(
                            tuple(ind), hs_derivative[None]), x1, x2, x3, self._r)
                    if compute_txt:
                        self.e_field_text += self._single_term_multipole_txt(ind,
                                                                             current_moment,
                                                                             "dh_dt")

        if self.verbose:
            print("Done.")

        return self.e_field

    @cython.ccall
    def compute_b_field(
        self,
        x1: ndarray,
        x2: ndarray,
        x3: ndarray,
        t: ndarray,
        h_sym,
        t_sym,
        verbose=False,
        delayed=True,
        compute_grid=True,
        compute_txt=False,
        shift=0,
    ):
        """
        Compute the magnetic field B from the moments. The method `recurse()` and `set_moments(...)` must be run
        beforehand.

        :return: the magnetic field as a 5-dimensional array. If :compute_grid: is True, the dimensions correspond to
         (dimension, x1, x2, x3, t), otherwise, (dimension, x, t).

        See the method :compute_e_field: for a full description of the arguments.
        """
        x1, x2, x3, t, moment_shape, hs_0 = self._prepare_arguments(
            x1, x2, x3, t, h_sym, t_sym,
            verbose=verbose,
            delayed=delayed,
            compute_grid=compute_grid,
            shift=shift,
            magnetic=True
        )

        for ind, _ in np.ndenumerate(zeros(self._shape)):
            ind = np.array(ind)
            if np_sum(ind) <= self.max_order:
                a1, a2, a3 = ind
                if self.verbose:
                    sys.stdout.write("\rComputing index {}...".format(ind))
                magnetic_moment = self.mu * self.magnetic_moment[:, a1, a2, a3].reshape(moment_shape)
                if np.any(magnetic_moment) > self.thresh:
                    self.b_field += self._single_term_multipole(ind,
                                                                magnetic_moment,
                                                                hs_0.get(tuple(ind), hs_0[None]),
                                                                x1, x2, x3, self._r)
                    if compute_txt:
                        self.b_field_text += self._single_term_multipole_txt(ind,
                                                                             magnetic_moment,
                                                                             "dh_dt")

        if self.verbose:
            print("Done.")

        return self.b_field

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
                fun = sympy.lambdify(t_sym, hs_sym[order])
                if self._causal:
                    hs[order] = fun(t - self._r / self.c)
                else:
                    hs[order] = -fun(t + self._r / self.c)
            else:
                hs[order] = sympy.lambdify(t_sym, hs_sym[order])(t)

        return self._repack_hs(hs)

    def _repack_hs(self, hs):
        hs_integral = list()
        hs_derivative = list()
        hs_0 = [hs[order] for order in range(0, self.max_order + 3)]
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

        return hs_0, hs_derivative, hs_integral

    def _handle_h_dict(self, h, t, shift=0):
        hs_0, hs_derivative, hs_integral = (dict(), ) * 3
        for key in h:
            hs_0[key], hs_derivative[key], hs_integral[key] = self._handle_h_array(h[key], t, shift=shift)
        return hs_0, hs_derivative, hs_integral

    def _handle_h_array(self, h, t, shift=0):

        dt = np.max(np.diff(t))

        try:
            h = np.array(h)
        except ValueError:
            h_np = np.zeros((3, t.size))
            for dim in range(3):
                h_np[dim, :] = h[dim]
            h = h_np

        def integrate_array(x):
            return np.cumsum(x, axis=-1) * dt

        def derivative(x):
            return np.gradient(x, dt, axis=-1)

        hs = {shift: h}
        for integral_order in range(shift - 1, -1 - 1, -1):
            hs[integral_order] = integrate_array(hs[integral_order + 1])

        for i in range(1 + shift, self.max_order + 3):
            hs[i] = derivative(hs[i - 1])

        h_sym_callable = dict()
        kwargs = {"bounds_error": False, "fill_value": 0.}
        for order in hs:
            if self.delayed:
                h_sym_callable[order] = interp1d(
                    t.squeeze(), hs[order], **kwargs
                )(t - self._r / self.c)
            else:
                h_sym_callable[order] = interp1d(t.squeeze(), hs[order], **kwargs)(t)

        return self._repack_hs(h_sym_callable)

    def _single_term_multipole(self,
                               ind: ndarray,
                               moment: ndarray,
                               hs: list | dict,
                               *args):
        return (-1) ** np_sum(ind) / fact(ind) \
               * moment * self._evaluate(tuple(ind), *args, hs) / 4 / np.pi

    def _single_term_multipole_txt(self,
                                   ind: ndarray,
                                   moment: ndarray,
                                   hs: str):
        return f"""  {(-1) ** np_sum(ind):+d}/{fact(ind)}"""\
               f"""*{list(map('{:.2e}'.format, moment.flatten()))}*{self._evaluate_txt(tuple(ind), hs)}/(4pi)\n"""

    def __repr__(self) -> str:
        return f"Solution: max_order={self.max_order}, c={self.c}, ran_recurse={self.ran_recurse}"

    def get_e_field_text(self) -> str:
        """
        Get a text description of the electric field. Returns an empty string if the method :compute_e_field: has not
        yet been called.

        :return: a human-readable description of the electric field
        """
        return self.e_field_text

    def get_b_field_text(self) -> str:
        """
        Get a text description of the magnetic field. Returns an empty string if the method :compute_b_field: has not
        yet been called.

        :return: a human-readable description of the magnetic field
        """
        return self.b_field_text


def fact(a) -> numbers.Number:
    """
    Compute the factorial of a multi-index
    
    :param a: the multi-index (an iterable) 
    :return: the factorial of a (i.e., a1!a2!...)
    """
    res: numbers.Number = 1
    for i in a:
        res *= factorial(i)
    return res


def set_extremities(x: np.ndarray, ratio: float, dim: int = 0, val: float = 0) -> np.ndarray:
    """
    Set the extremities of an array to a constant value
    
    :param x: array whose extremities will be set
    :param ratio: proportion of values to set, 0 <= ratio <= 1
    :param dim: the dimension along which the value is set
    :param val: the value to set

    :raises ValueError: when the ratio is not between 0 and 1
    :rtype: np.ndarray
    """
    if ratio < 0 or ratio > 1:
        raise ValueError(f"Expected a ratio between 0 and 1, got {ratio}")

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
    s = Solution(max_order=3)
    s.recurse()
    s.set_moments(
        current_moment=lambda a1, a2, a3: [1., 0., 0.] if a1 == a2 == a3 == 0 or a1 == 1 and a2 == a3 == 0
        else [0., 0., 0.])
    t = np.linspace(0, 10, 100)
    x1 = np.array([.6, ])
    x2, x3 = x1.copy(), x1.copy()
    h = {(0, 0, 0): 1 * np.exp(-t), (1, 0, 0): -2 * np.exp(-t)}
    # h = np.exp(-t)
    e = s.compute_e_field(x1, x2, x3, t, h, None, delayed=False)
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    plt.plot(t, e[0, 0, 0, 0, :])
    plt.show()
