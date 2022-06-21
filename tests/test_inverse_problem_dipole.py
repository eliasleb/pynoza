import pynoza
import numpy as np
import pytest
import matplotlib.pyplot as plt
import inverse_problem
import scipy


def test_charge_moment_computation():
    # Moment of a dipole
    current_moment = np.zeros((3, 3, 3, 3))
    current_moment[0, 0, 0, 2] = 1
    charge_moment = inverse_problem.get_charge_moment(current_moment)
    charge_moment_analytical = np.zeros((3, 3, 3, 3))
    charge_moment_analytical[0, 0, 2, 2] = 2

    for i in {0, 1}:
        a = [0, 0, 1]
        a[i] = 1
        charge_moment_analytical[a[0], a[1], a[2], i] = 1
    print(f"{charge_moment=}")
    print(f"{charge_moment_analytical=}")
    print(np.any(charge_moment != 0), np.any(charge_moment_analytical != 0))

    assert np.all(charge_moment == charge_moment_analytical)

    if __name__ == "__main__":
        plt.ion()
        inverse_problem.plot_moment(current_moment)
        inverse_problem.plot_moment(charge_moment)
        inverse_problem.plot_moment(charge_moment_analytical)

        plt.pause(0.1)
        plt.show()
        input("Press Enter to continue...")


@pytest.mark.skip(reason="Not done yet")
def test_inverse_problem_simple():
    """
    Test the pynoza :solution: class by solving inverse problem on a dipole source
    """
    x1 = np.array([-1, 1])
    x2 = x1.copy()
    x3 = x1.copy()

    t = np.linspace(0, 20, 100)
    c0 = 3e8
    wavelength = 1
    f = 1 / wavelength
    gamma = np.sqrt(12/7)/f
    t0 = 5*gamma

    h_true = np.exp(-((t-t0)/gamma)**2) * (4 * ((t-t0)/gamma)**2 - 2)

    dt = np.max(np.diff(t))
    f = np.linspace(0, 1/dt, t.size)


    def get_h_num(h, t):
        h[-h.size//3:] = 0
        h[0] = 0
        return scipy.interpolate.interp1d(np.linspace(t.min(), t.max(), h.size),
                                          h,
                                          kind="cubic")(t)

    kwargs = {"tol": 1e-10,
              "n_points": 20,
              "error_tol": 5e-2,
              "coeff_derivative": 0,
              "verbose_every": 2,
              "plot": __name__ == "__main__",
              "h_num": get_h_num,
              "find_center": True,
              "scale": 1}

    order = 2

    shape_mom = (order + 1, order + 1, order + 1, 3)
    dim_mom = 3

    def get_current_moment(moment):
        current_moment = np.zeros(shape_mom)
        current_moment[0, 0, 0, :] = moment
        return current_moment

    e_true = direct_problem_simple(x1, x2, x3, t, h_true, order=order)
    current_moment, h, center = inverse_problem.inverse_problem(order, e_true, x1, x2, x3, t,
                                                                get_current_moment, dim_mom, **kwargs)

    if __name__ == "__main__":

        plt.ion()
        print(f"{center=}")

        inverse_problem.plot_moment(current_moment)
        inverse_problem.plot_moment(inverse_problem.get_charge_moment(current_moment))

        plt.figure()
        h -= h[0]
        plt.plot(t, h/np.max(np.abs(h)))
        h_max = np.max(np.abs(h_true))
        plt.plot(t, h_true/h_max, "--")
        plt.xlabel("Time (relative)")
        plt.ylabel("Amplitude (normalized)")
        plt.legend(["Inverse problem", "True solution"])
        plt.title("Current vs time")
        plt.pause(0.1)
        plt.show()
        input("Press Enter to continue...")


def direct_problem_simple(x1, x2, x3, t, h, order=2):

    def current_moment(a1, a2, a3):
        if a1 == 0 and a2 == 0 and a3 == 0:
            return [0, 0, 1]
        else:
            return [0, 0, 0]

    def charge_moment(a1, a2, a3):
        a = (a1, a2, a3)
        if a == (1, 0, 1):
            return [1, 0, 0]
        elif a == (0, 1, 1):
            return [0, 1, 0]
        elif a == (0, 0, 2):
            return [0, 0, 2]
        else:
            return [0, 0, 0]

    sol = pynoza.solution.Solution(max_order=order)
    sol.recurse()
    sol.set_moments(charge_moment=charge_moment, current_moment=current_moment)
    e = sol.compute_e_field(x1, x2, x3, t, h, None)
    return e


if __name__ == "__main__":
#    test_charge_moment_computation()
    test_inverse_problem_simple()
