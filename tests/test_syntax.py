#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pynoza
import pytest
import sympy
import numpy as np
import sympy


def check_args(fun, *args, **kwargs):
    with pytest.raises(ValueError):
        fun(*args, **kwargs)


def test_inputs():
    check_args(pynoza.Solution, wave_speed="hello you")
    check_args(pynoza.Solution, max_order=0.)
    s = pynoza.Solution()
    check_args(s.set_moments, charge_moment=1)
    check_args(s.set_moments, current_moment=1)
    check_args(s.set_moments, charge_moment=lambda a1, a2, a3: "hello you")
    check_args(s.set_moments, current_moment=lambda a1, a2, a3: "hello you")
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


if __name__ == "__main__":
    sol = pynoza.Solution(max_order=2)
    print(sol)
    sol.recurse()
    print(sol)
