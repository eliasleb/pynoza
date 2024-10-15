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
from itertools import product as itertools_product


@cython.ccall
def levi_civita(i: int, j: int, k: int, start_at_0=True):
    if start_at_0:
        i, j, k = i + 1, j + 1, k + 1
    ijk = (i, j, k)
    if ijk in ((1, 2, 3), (2, 3, 1), (3, 1, 2)):
        return 1
    elif ijk in ((3, 2, 1), (1, 3, 2), (2, 1, 3)):
        return -1
    return 0


@cython.ccall
def get_charge_moment(current_moment: ndarray, return_mapping=False):
    """
    Compute a charge moment that is compatible with the conservation of charge

    :param current_moment: an array with the current moments
    :param return_mapping: return a dict to keep a track of which charge moments correspond to which current moments
    :return: the corresponding charge moment (+ eventually the charge-current mapping)

    The moment at index `(i, a1, a2, a3)` is the moment of the `i`-th coordinate corresponding to the multi-index
    `(a1, a2, a3)`
    """
    charge_moment = np.zeros(current_moment.shape)
    mapping = {}
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
                    if tuple(ind) not in mapping:
                        mapping[tuple(ind)] = dict()
                    mapping[tuple(ind)][(j, ) + tuple(b)] = a[j] * (a[j] - 1)
            else:
                b[i] -= 1
                b[j] -= 1
                if a[j] >= 1 and a[i] >= 1:
                    charge_moment[i, a1, a2, a3] += a[j] * a[i] \
                        * current_moment[j, b[0], b[1], b[2]]
                    if tuple(ind) not in mapping:
                        mapping[tuple(ind)] = dict()
                    mapping[tuple(ind)][(j, ) + tuple(b)] = a[j] * a[i]
    charge_moment = -charge_moment
    if return_mapping:
        return charge_moment, mapping
    return charge_moment


@cython.ccall
def get_magnetic_moment(current_moment: ndarray, return_mapping=False):
    """
    Compute the magnetic moment corresponding to the curl of the current density from the current moments.

    :param current_moment: an array with the current moments
    :param return_mapping: return a dict to keep a track of which current moments correspond to which magnetic moments
    :return: the corresponding magnetic moment (+ eventually the charge-current mapping)
    :rtype: ndarray

    The moment at index `(i, a1, a2, a3)` is the moment of the `i`-th coordinate corresponding to the multi-index
    `(a1, a2, a3)`
    """
    magnetic_moment = np.zeros(current_moment.shape)
    mapping = {}
    for ind, _ in np.ndenumerate(magnetic_moment):
        i, a1, a2, a3 = ind
        a = [a1, a2, a3]
        for j, k in itertools_product(range(3), repeat=2):
            lc = levi_civita(i, j, k)
            if a[j] > 0 and lc != 0:
                a_copy = [a1, a2, a3]
                a_copy[j] -= 1
                magnetic_moment[i, a1, a2, a3] += a[j] * lc * current_moment[k, a_copy[0], a_copy[1], a_copy[2]]
                if tuple(ind) not in mapping:
                    mapping[tuple(ind)] = dict()
                mapping[tuple(ind)][(k, ) + tuple(a_copy)] = a[j] * lc

    magnetic_moment = -magnetic_moment
    if return_mapping:
        return magnetic_moment, mapping
    return magnetic_moment


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


def check_fun_type(fun):
    if isinstance(fun, ndarray) or isinstance(fun, sympy.Expr):
        return
    raise ValueError(f"The time-dependent excitation :{str(fun):.20}...: must be either a NumPy array or  "
                     "a SymPy expression")


@cython.ccall
def _process_moment_mapping(mapping, h0_dict):
    h_tfm_dict = dict()
    for dim_and_multi_index, current_mapping in mapping.items():
        if dim_and_multi_index[1:] not in h_tfm_dict:
            h_tfm_dict[dim_and_multi_index[1:]] = np.zeros(next(iter(h0_dict.values())).shape)
        for current_dim_and_multi_index, coefficient in current_mapping.items():
            if current_dim_and_multi_index[1:] in h0_dict:
                h_tfm_dict[dim_and_multi_index[1:]][:, dim_and_multi_index[0], ...] += \
                    coefficient * h0_dict[current_dim_and_multi_index[1:]][:, current_dim_and_multi_index[0], ...]
    return h_tfm_dict


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
    _all_multi_indices: set

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
        self._all_multi_indices = set()

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
            # if signature[R] > 0:
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
            if np.sum(ind) <= self.max_order:
                self.current_moment[:, ind[0], ind[1], ind[2]] = current_moment(*ind)
                if charge_moment is not None:
                    self.charge_moment[:, ind[0], ind[1], ind[2]] = charge_moment(*ind)
        if charge_moment is None:
            self.charge_moment, self._rho_to_j_mapping = get_charge_moment(
                self.current_moment, return_mapping=True)
        self.magnetic_moment = get_magnetic_moment(self.current_moment)

    @cython.ccall
    def _prepare_arguments(
            self,
            x1: ndarray,
            x2: ndarray,
            x3: ndarray,
            t: ndarray,
            h,
            t_sym,
            verbose,
            delayed,
            compute_grid,
            shift,
            magnetic
    ):
        if not self.ran_recurse:
            raise RuntimeError("You must first run the `recurse' method.")

        if isinstance(h, dict):
            a_key = next(iter(h.keys()))
            if isinstance(a_key, tuple):
                if len(a_key) != 3:
                    raise ValueError("When h_sym is a dictionary, the keys must be tuples of length 3")
                a_value = next(iter(h.values()))
                if not isinstance(a_value, list):
                    raise ValueError("Values of the dictionary :h: must be lists")
                if len(a_value) != 3:
                    raise ValueError("Values of the dictionary :h: must be lists of length 3")
                check_fun_type(a_value[0])
            else:
                raise ValueError("When h_sym is a dictionary, the keys must be the tuples (ax, ay, az)")
            if self.ran_set_moments:
                raise ValueError(f"When providing :h: as a dictionary, the moments set using :set_moments(...):"
                " are ignored")
        else:
            check_fun_type(h)
            if not self.ran_set_moments:
                raise RuntimeError("When providing a single function in :h:, you must first run the method"
                                   " :set_moments(...):")

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
            moment_shape = (1, 3, 1, 1, 1, 1)  # derivative, dimension, x1, x2, x3, t
            _array = zeros((3, x1.size, x2.size, x3.size, t.size))
        else:
            x1 = x1.reshape((x1.size, 1))
            x2 = x2.reshape((x2.size, 1))
            x3 = x3.reshape((x3.size, 1))
            t = t.reshape((1, t.size))
            moment_shape = (1, 3, 1, 1)  # derivative, dimension, position, t
            _array = zeros((3, x1.size, t.size))

        if magnetic:
            self.b_field = _array
        else:
            self.e_field = _array

        self._r = np.sqrt(x1 ** 2 + x2 ** 2 + x3 ** 2)

        if isinstance(h, ndarray):
            hs_0, hs_derivative, hs_integral = self._handle_h_array(h, t, shift=shift)
        elif isinstance(h, dict):
            hs_0, hs_derivative, hs_integral = self._handle_h_dict(h, t, shift=shift)
        else:
            hs_0, hs_derivative, hs_integral = self._handle_h_symbolic(h, t_sym, t)

        hs_0_dict, hs_derivative_dict, hs_integral_dict = dict(), dict(), dict()

        if self.ran_set_moments:
            # Backward-compatibility: transform the combination "single function" + "moments" into a unified dictionary
            # embedding the moments and the function
            for ind, _ in np.ndenumerate(zeros(self._shape)):
                # if np_sum(ind) > self.max_order:
                #     continue
                moment_slice = (slice(None), ) + ind
                if magnetic:
                    if np.any(np.abs(self.magnetic_moment[moment_slice]) > self.thresh):
                        hs_0_dict[ind] = self.magnetic_moment[moment_slice].reshape(moment_shape) * hs_0[:, None, ...]
                else:
                    if np.any(np.abs(self.current_moment[moment_slice]) > self.thresh):
                        hs_derivative_dict[ind] = self.current_moment[moment_slice].reshape(moment_shape) \
                                                  * hs_derivative[:, None, ...]
                    if np.any(np.abs(self.charge_moment[moment_slice]) > self.thresh):
                        hs_integral_dict[ind] = self.charge_moment[moment_slice].reshape(moment_shape) \
                                                  * hs_integral[:, None, ...]
        else:
            # We must first do current-charge and current-magnetic-current mappings
            if magnetic:
                hs_0_dict = _process_moment_mapping(
                    get_magnetic_moment(np.zeros((3,) + (self.max_order + 1,) * 3), return_mapping=True)[1],
                    hs_0
                )
            else:
                hs_integral_dict = _process_moment_mapping(
                    get_charge_moment(np.zeros((3,) + (self.max_order + 1,) * 3), return_mapping=True)[1],
                    hs_integral
                )

        hs_0, hs_derivative, hs_integral = hs_0_dict, hs_derivative_dict, hs_integral_dict

        if magnetic:
            self._all_multi_indices = set(hs_0.keys())
            return x1, x2, x3, t, moment_shape, hs_0
        self._all_multi_indices = set(hs_derivative.keys()).union(set(hs_integral.keys()))
        return x1, x2, x3, t, moment_shape, hs_derivative, hs_integral

    @cython.ccall
    def compute_e_field(
        self,
        x1: ndarray,
        x2: ndarray,
        x3: ndarray,
        t: ndarray,
        h,
        t_sym=None,
        verbose=False,
        delayed=True,
        compute_grid=True,
        compute_txt=False,
        shift=0,
    ):
        """
        Compute the electric field from the moments. The method `recurse()` must be run beforehand.
        If separability is assumed, the method `set_moments(...)` must also  be run beforehand. In this case, the
        time-dependent excitation is assumed to be a function. Otherwise, the moments are embedded in the time-dependent
        excitation :h: (see below).

        :return: the electric field as a 5-dimensional array. If :compute_grid: is True, the dimensions correspond to
         (dimension, x1, x2, x3, t), otherwise, (dimension, x, t).
        :param x1: array of the spatial coordinates to evaluate the e_field-field at (aka x)
        :param x2: array of the spatial coordinates to evaluate the e_field-field at (aka y)
        :param x3: array of the spatial coordinates to evaluate the e_field-field at (aka z)
        :param t: array of the time coordinates to evaluate the e_field-field at
        :param h: time-dependent excitation. If separability is assumed, it is:
            - either a symbolic expression of :t_sym:
            - or a 1D numpy array of samples, of the same size as :t:.
            If separability is not assumed, meaning each moment is permitted to exhibit distinct time-dependences, then
            :h: must be a dictionary following:
                (a1, a2, a3) -> [f1, f2, f3]
            where (a1, a2, a3) is the multi-index of a given moment, and f1, f2, f3 are the corresponding time-dependent
            excitations, for the first to last dimensions. f1, f2, f3 must be as above: either a symbolic expression of
            :t_sym:, or a 1D numpy array of samples the same size as :t:.
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
            x1, x2, x3, t, h, t_sym,
            verbose=verbose,
            delayed=delayed,
            compute_grid=compute_grid,
            shift=shift,
            magnetic=False
        )

        for multi_index in self._all_multi_indices:
            if np_sum(multi_index) <= self.max_order:
                if multi_index in hs_integral:
                    self.e_field += self._single_term_multipole(multi_index,
                                                                -self.mu * self.c ** 2 * hs_integral[multi_index],
                                                                x1, x2, x3, self._r)
                    if compute_txt:
                        self.e_field_text += self._single_term_multipole_txt(
                            multi_index, "rho", "int_h")

                if multi_index in hs_derivative:
                    self.e_field += self._single_term_multipole(multi_index,
                                                                -self.mu * hs_derivative[multi_index],
                                                                x1, x2, x3, self._r)
                    if compute_txt:
                        self.e_field_text += self._single_term_multipole_txt(
                            multi_index, "J", "dh_dt")

        return self.e_field

    @cython.ccall
    def compute_b_field(
        self,
        x1: ndarray,
        x2: ndarray,
        x3: ndarray,
        t: ndarray,
        h,
        t_sym=None,
        verbose=False,
        delayed=True,
        compute_grid=True,
        compute_txt=False,
        shift=0,
    ):
        """
        Compute the magnetic field B from the moments. The method `recurse()` must be run beforehand.

        :return: the magnetic field as a 5-dimensional array. If :compute_grid: is True, the dimensions correspond to
         (dimension, x1, x2, x3, t), otherwise, (dimension, x, t).

        See the method :compute_e_field: for a full description of the arguments.
        """
        x1, x2, x3, t, moment_shape, hs_0 = self._prepare_arguments(
            x1, x2, x3, t, h, t_sym,
            verbose=verbose,
            delayed=delayed,
            compute_grid=compute_grid,
            shift=shift,
            magnetic=True
        )

        for ind in self._all_multi_indices:
            if np_sum(ind) <= self.max_order:
                self.b_field += self._single_term_multipole(ind,
                                                            self.mu * hs_0[ind],
                                                            x1, x2, x3, self._r)
                if compute_txt:
                    self.b_field_text += self._single_term_multipole_txt(ind,
                                                                         "nabla_x_j",
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
                    hs_derivative.append(np.zeros(hs[0].shape))
                    hs_integral.append(np.zeros(hs[0].shape))
            else:
                hs_integral.append(hs[order])

        return _to_ndarray_safer(hs_0), _to_ndarray_safer(hs_derivative), _to_ndarray_safer(hs_integral)

    def _handle_h_dict(self, h, t, shift=0):
        hs_0, hs_derivative, hs_integral = dict(), dict(), dict()
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
                               hs,
                               *args):
        return (-1) ** np_sum(ind) / fact(ind) \
               * self._evaluate(tuple(ind), *args, hs) / 4 / np.pi

    def _single_term_multipole_txt(self,
                                   ind: ndarray,
                                   moment_name: str,
                                   hs: str):
        return f"""  {(-1) ** np_sum(ind):+d}/{fact(ind)} C_{moment_name}({ind})"""\
               f"""*{self._evaluate_txt(tuple(ind), hs)}/(4pi)\n"""

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


@cython.ccall
def _to_ndarray_safer(a: list):
    shape = None
    for i, ai in enumerate(a):
        if isinstance(ai, ndarray):
            shape = ai.shape
        else:
            if shape is None:
                raise RuntimeError(f"At least one element of '{a}' should be a NumPy array")
            a[i] = ai * np.ones(shape)
    return np.array(a)


if __name__ == "__main__":

    def small_test():
        s = Solution(max_order=3)
        s.recurse()
        s.set_moments(
            current_moment=lambda a1, a2, a3: [1., 0., 0.] if a1 == a2 == a3 == 0 or a1 == 1 and a2 == a3 == 0
            else [0., 0., 0.])
        t = np.linspace(0, 10, 100)
        x1 = np.array([.6, ])
        x2, x3 = x1.copy(), x1.copy()
        # h = {(0, 0, 0): [2 * np.exp(-t), 0 * t, 0 * t], (1, 0, 0): [.000001 * np.exp(-t), 0 * t, 0 * t]}
        h = np.exp(-t)
        s.compute_e_field(x1, x2, x3, t, h, None, delayed=False)
        s.compute_b_field(x1, x2, x3, t, h, None, delayed=False)

        s = Solution(max_order=3)
        s.recurse()
        t = np.linspace(0, 10, 100)
        x1 = np.array([.6, ])
        x2, x3 = x1.copy(), x1.copy()
        h = {(0, 0, 0): [2 * np.exp(-t), 0, 0], (1, 0, 0): [-.1 * np.exp(0.001 * t), 0, 0]}
        e = s.compute_e_field(x1, x2, x3, t, h, None, delayed=False)
        b = s.compute_b_field(x1, x2, x3, t, h)

        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        plt.plot(t, e[0, 0, 0, 0, :])
        plt.show()

    small_test()
