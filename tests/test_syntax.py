#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pynoza
import pytest


def check_args(fun,*args,**kwargs):
    with pytest.raises(ValueError):
        fun(*args,**kwargs)

def test_inputs():
    check_args(pynoza.Solution, wave_speed="hello you")
    check_args(pynoza.Solution, max_order=0.)
    s = pynoza.Solution()
    check_args(s.set_moments, charge_moment=1)
    check_args(s.set_moments, current_moment=1)
    check_args(s.set_moments, charge_moment=lambda a1, a2, a3: "hello you")
    check_args(s.set_moments, current_moment=lambda a1, a2, a3: "hello you")
    with pytest.raises(RuntimeError):
        s.compute_e_field(0,
                          0,
                          0,
                          0,
                          0,
                          0)



