import pynoza
import numpy as np
import pytest
import matplotlib.pyplot as plt
import inverse_problem
import sympy


@pytest.mark.skip(reason="Not done yet")
def test_inverse_problem_simple():
    """
    Test the pynoza :solution: class by solving inverse problem on a dipole source
    """
    x1 = np.array([-3, 3])
    x2 = x1.copy()
    x3 = x1.copy()

    t = np.linspace(0, 20, 100)
    c0 = 3e8
    t1 = 1e-9*c0
    t2 = 5e-9*c0

    h_true = (1-np.exp(-t/t1))*(t < t2) + np.exp(-(t - t2)/t2)*(t >= t2)

    dt = np.max(np.diff(t))
    f = np.linspace(0, 1/dt, t.size)

    e_true = direct_problem_simple(x1, x2, x3, t, h_true)
    kwargs = {"tol": 1e-8,
              "n_points": 20,
              "error_tol": 0.5e-2,
              "coeff_derivative": 0}
    current_moment, charge_moment, h = inverse_problem.inverse_problem(1, e_true, x1, x2, x3, t, **kwargs)

    h -= h[0]
    plt.plot(t, h/np.max(np.abs(h)))
    h_max = np.max(np.abs(h_true))
    plt.plot(t, h_true/h_max, "--")
    plt.xlabel("Time (relative)")
    plt.ylabel("Amplitude (normalized)")
    plt.legend(["Inverse problem", "True solution"])
    plt.title("Current vs time")
    plt.show()

def direct_problem_simple(x1, x2, x3, t, h):

    def current_moment(a1, a2, a3):
        if a1 == 0 and a2 == 0 and a3 == 0:
            return [0, 0, 1]
        else:
            return [0, 0, 0]

    def charge_moment(a1, a2, a3):
        if a3 == 1:
            if a1 == 1 and a2 == 0:
                return [1, 0, 0]
            elif a1 == 0 and a2 == 1:
                return [0, 1, 0]
            elif a1 == 0 and a2 == 0:
                return [0, 0, 1]
            else:
                return [0, 0, 0]
        else:
            return [0, 0, 0]

    sol = pynoza.solution.Solution(max_order=1)
    sol.recurse()
    sol.set_moments(charge_moment=charge_moment, current_moment=current_moment)
    e = sol.compute_e_field(x1, x2, x3, t, h, None)
    return e


if __name__ == "__main__":
    test_inverse_problem_simple()
