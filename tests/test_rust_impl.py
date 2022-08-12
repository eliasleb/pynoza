import speenoza
import numpy as np
import pynoza
import matplotlib.pyplot as plt


def test_syntax():
    max_order = 2
    x1 = np.array([1, 0, 0, ], dtype=float)
    x2 = np.array([0, 1, 0, ], dtype=float)
    x3 = np.array([0, 0, 1, ], dtype=float)
    t = np.linspace(0, 10, 100, dtype=float)
    h = np.sin(2 * np.pi * t)
    moment = np.zeros((3, max_order + 1, max_order + 1, max_order + 1, ))
    moment[2, 0, 0, 0] = 1.
    sol = speenoza.multipole_e_field(x1, x2, x3, t, h, moment)



if __name__ == "__main__":
    test_syntax()
