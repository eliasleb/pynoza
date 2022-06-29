#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 11:16:58 2021

@author: Elias Le Boudec, elias.leboudec@epfl.ch
"""
import numpy as np
import sys
import sympy
import numbers
import typing


X1: int = 0
X2: int = 1
X3: int = 2
H: int = 3
R: int = 4


class Solution:
    """A class to compute solutions of Maxwell's equations, based on"""
    """time-domain multipole moments."""

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
        self._shape: tuple[int, int, int] = (max_order + 1, max_order + 1, max_order + 1)
        self._aux_func: dict[tuple[int, int, int], dict[tuple[int, int, int, int, int], float]] = dict()
        for ind, _ in np.ndenumerate(np.zeros(self._shape)):
            self._aux_func[tuple(ind)] = dict()
        self._aux_func[(0, 0, 0)] = {(0, 0, 0, 0, 1): 1.}
        self.mu: float = 4 * np.pi * 1e-7

        self.ran_recurse: bool = False
        self.ran_set_moments: bool = False
        self.current_moment: typing.Callable[[int, int, int], numbers.Number] = lambda a1, a2, a3: 0
        self.charge_moment: typing.Callable[[int, int, int], numbers.Number] = lambda a1, a2, a3: 0
        self.verbose: bool = False
        self.delayed: bool = True

        self.e_field: np.ndarray = np.array([])

        self.e_field_text: str = ""

    def _increase_order(self, known_index: tuple[int, int, int], index: tuple[int, int, int]) -> None:
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
                identity_first_term: list[int, int, int, int, int] = list(signature)
                identity_first_term[known_dim] -= 1  # differentiate
                try:
                    self._aux_func[index][tuple(identity_first_term)] \
                        += coefficient * exponent_x_i
                except KeyError:
                    self._aux_func[index][tuple(identity_first_term)] \
                        = coefficient * exponent_x_i
            exponent_r: int = signature[R]
            identity_second_term: list[int, int, int, int, int] = list(signature)
            identity_second_term[known_dim] += 1  # numerator
            identity_second_term[R] += 2  # denominator
            try:
                self._aux_func[index][tuple(identity_second_term)] -= \
                    coefficient * exponent_r
            except KeyError:
                self._aux_func[index][tuple(identity_second_term)] = \
                    -coefficient * exponent_r
            # Time-derivative term
            identity_third_term: list[int, int, int, int, int] = list(signature)
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
            for ind, _ in np.ndenumerate(np.zeros(self._shape)):
                ind: np.ndarray = np.array(ind)
                if np.sum(ind) == order:
                    known_ind: np.array = ind.copy()
                    known_ind[np.where(ind > 0)[0][0]] -= 1
                    if verbose:
                        sys.stdout.write("\rComputing order {}...".format(order))
                    self._increase_order(tuple(known_ind), tuple(ind))
        if verbose:
            print("Done.")

    def _evaluate(self,
                  ind: tuple[int, int, int],
                  t: np.ndarray,
                  x1: np.ndarray,
                  x2: np.ndarray,
                  x3: np.ndarray,
                  r: np.ndarray,
                  hs: list[typing.Callable[[np.ndarray], np.ndarray]],
                  **_) -> np.ndarray:
        """Evaluate the auxiliary function.
        
        Positional arguments:
        ind -- multi-index of the auxiliary function
        T -- evaluated time
        X1 -- evaluated first coordinate (aka x)
        X2 -- evaluated second coordinate (aka y)
        X3 -- evaluated third coordinate (aka z)
        R -- equal to X1**2+X2**2+X3**2. Passed to avoid computing it repeatedly.
        Hs -- dictionary of the derivatives of the time-dependent excitation function.
              Must be in the form {order:derivative of order} for order=-1..max_order+2
        """
        y: np.ndarray = np.zeros(x1.shape)
        dy: np.ndarray = np.zeros(x1.shape)
        for signature in self._aux_func[ind]:
            if self.delayed:
                dy = hs[signature[H]](t - r / self.c) * self._aux_func[ind][signature]
            else:
                dy = hs[signature[H]](t) * self._aux_func[ind][signature]
            if signature[X1] > 0:
                dy *= x1 ** signature[X1]
            if signature[X2] > 0:
                dy *= x2 ** signature[X2]
            if signature[X3] > 0:
                dy *= x3 ** signature[X3]
            if signature[R] > 0:
                dy /= r ** signature[R]
            y += dy
        return y

    def _evaluate_txt(self,
                      ind: tuple[int, int, int],
                      h: str) -> str:
        """Evaluate the auxiliary function as a symbolic expression

            :param ind: multi-index to evaluate at
            :param h: name of the function
            :return: a string describing the auxiliary function
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
                    current_moment: typing.Callable[[int, int, int], list[numbers.Number,
                                                                          numbers.Number,
                                                                          numbers.Number]] = lambda a1, a2, a3: [0,
                                                                                                                 0,
                                                                                                                 0],
                    charge_moment: typing.Callable[[int, int, int], list[numbers.Number,
                                                                         numbers.Number,
                                                                         numbers.Number]] = lambda a1, a2, a3: [0,
                                                                                                                0,
                                                                                                                0],)\
            -> None:
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
        self.current_moment = current_moment
        self.charge_moment = charge_moment

    def compute_e_field(self,
                        x1: np.ndarray,
                        x2: np.ndarray,
                        x3: np.ndarray,
                        t: np.ndarray,
                        h_sym,
                        t_sym,
                        **kwargs) -> np.ndarray:
        """Compute the electric field from the moments.
        
        Positional arguments:
        x1,x2,x3 -- arrays of the spatial coordinates to evaluate the e_field-field at (aka x,y,z)
        t -- array of the time coordinates to evaluate the e_field-field at
        h_sym -- symbolic time-dependent (:t_sym:) function describing the shape of the current
        t_sym -- symbolic variable representing time, used in h_sym
        
        Keyword arguments:
        verbose -- whether to display the progress (default False)
        delayed -- whether to evaluate the field at the retarded time t-r/c (default True)"""

        if not self.ran_recurse or not self.ran_set_moments:
            raise RuntimeError("You must first run the `recurse' and `set_moments' methods.")

        self.verbose = kwargs.pop("verbose", False)
        self.delayed = kwargs.pop("delayed", True)

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

        if kwargs:
            raise ValueError(f"Unexpected keyword arguments: {kwargs}")

        self.e_field = np.zeros((3, x1.size, x2.size, x3.size, t.size))
        self.e_field_text = ""

        x1, x2, x3, t = np.meshgrid(x1, x2, x3, t)
        r = np.sqrt(x1 ** 2 + x2 ** 2 + x3 ** 2)

        thresh = 1e-14

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
            hs[order] = sympy.lambdify(t_sym, hs_sym[order])

        hs_integral = list()
        hs_derivative = list()
        for order in range(-1, self.max_order + 3):
            if order > 0:
                if order <= self.max_order:
                    hs_derivative.append(hs[order])
                    hs_integral.append(hs[order])
                else:
                    hs_derivative.append(lambda _: 0)
                    hs_integral.append(lambda _: 0)
            else:
                hs_integral.append(hs[order])

        for ind, _ in np.ndenumerate(np.zeros(self._shape)):
            ind = np.array(ind)
            if np.sum(ind) <= self.max_order:
                a1, a2, a3 = ind
                if self.verbose:
                    sys.stdout.write("\rComputing index {}...".format(ind))
                charge_moment = -self.mu * self.c ** 2 * np.array(
                    self.charge_moment(a1, a2, a3)).reshape((3, 1, 1, 1, 1))
                current_moment = -self.mu * np.array(
                    self.current_moment(a1, a2, a3)).reshape((3, 1, 1, 1, 1))
                if np.any(charge_moment) > thresh:
                    self.e_field += self._single_term_multipole(ind,
                                                                charge_moment,
                                                                hs_integral,
                                                                t, x1, x2, x3, r,
                                                                **kwargs)
                    self.e_field_text += self._single_term_multipole_txt(ind,
                                                                         charge_moment,
                                                                         "int_h")
                if np.any(current_moment) > thresh:
                    self.e_field += self._single_term_multipole(ind,
                                                                current_moment,
                                                                hs_derivative,
                                                                t, x1, x2, x3, r,
                                                                **kwargs)
                    self.e_field_text += self._single_term_multipole_txt(ind,
                                                                         current_moment,
                                                                         "dhdt")

        if self.verbose:
            print("Done.")

        return self.e_field

    def _single_term_multipole(self,
                               ind: np.ndarray,
                               moment: np.ndarray,
                               hs: np.ndarray,
                               *args,
                               **kwargs):
        return (-1) ** np.sum(ind) / fact(ind) \
               * moment * self._evaluate(tuple(ind),
                                         *args,
                                         hs,
                                         **kwargs) / 4 / np.pi

    def _single_term_multipole_txt(self,
                                   ind: np.ndarray,
                                   moment: np.ndarray,
                                   hs: str):
        return f"""  {(-1) ** np.sum(ind):+d}/{fact(ind)}"""\
               f"""*{list(map('{:.2e}%'.format, moment.flatten()))}*{self._evaluate_txt(tuple(ind), hs)}/(4pi)\n"""

    def __repr__(self) -> str:
        return f"Solution: {self.max_order=}, {self.c=}, {self.ran_recurse=}"


def fact(a) -> numbers.Number:
    res: numbers.Number = 1
    for i in a:
        res *= np.math.factorial(i)
    return res


if __name__ == "__main__":
    pass
