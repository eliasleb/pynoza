#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022  Elias Le Boudec
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.


import pynoza
import pytest
import numpy as np
import sympy


def check_args(fun, *args, error=Exception, **kwargs):
    with pytest.raises(error):
        fun(*args, **kwargs)


def test_inputs():
    check_args(pynoza.Solution, wave_speed="hello", error=TypeError)
    check_args(pynoza.Solution, max_order=0., error=TypeError)
    s = pynoza.Solution()
    check_args(s.set_moments, charge_moment=1, error=TypeError)
    check_args(s.set_moments, current_moment=1, error=TypeError)
    check_args(s.set_moments, charge_moment=lambda a1, a2, a3: "hello", error=ValueError)
    check_args(s.set_moments, current_moment=lambda a1, a2, a3: "hello", error=ValueError)
    x = np.array([0, ])
    t_sym = sympy.Symbol("t")
    h_sym = t_sym
    with pytest.raises(RuntimeError):
        s.compute_e_field(x, x, x, x, h_sym, t_sym)
    s.recurse()
    s.set_moments()
    with pytest.raises(ValueError):
        s.compute_e_field(x, x, x, x, {-1: None}, t_sym)
    t_sym = sympy.Symbol("t")
    h_sym = t_sym
    s.compute_e_field(x, x, x, x, h_sym, t_sym)


def test_multi_time_dependent_moments():
    s = pynoza.Solution()
    s.recurse()
    s.set_moments(
        current_moment=lambda a1, a2, a3: [1., 0., 0.] if a1 == a2 == a3 == 0 or a1 == 1 and a2 == a3 == 0
        else [0., 0., 0.])
    t = np.linspace(0, 10, 100)
    x1 = np.array([1., ])
    x2, x3 = x1.copy(), x1.copy()
    h = {(0, 0, 0): np.exp(-t), (1, 0, 0): -2 * np.exp(-t)}
    _e = s.compute_e_field(x1, x2, x3, t, h, None)
    h = {(0, 0, 0): [np.exp(-t), 0, 0], (1, 0, 0): [-2 * np.exp(-t), 0, 0]}
    _e = s.compute_e_field(x1, x2, x3, t, h, None)


if __name__ == "__main__":
    sol = pynoza.Solution(max_order=2)
    print(sol)
    sol.recurse()
    print(sol)
