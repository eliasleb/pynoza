import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import inverse_problem
import itertools
import scipy.interpolate
from matplotlib import cm
import time

filename = "../../../git_ignore/GLOBALEM/meep-dipole_v1.csv"
data = pd.read_csv(filename,
                   delimiter=",",
                   header=1)

c0 = 3e8
f = 1e9/c0
gamma = np.sqrt(12 / 7) / f
t0 = 3 * gamma
dt = 0.0149896229


times_added = 7
x1_symmetry = np.zeros((x1.size * times_added))
x2_symmetry = np.zeros((x1.size * times_added))
x3_symmetry = np.zeros((x1.size * times_added))
ex_symmetry = np.zeros((x1.size * times_added, ex.shape[1]))
ey_symmetry = np.zeros((x1.size * times_added, ex.shape[1]))
ez_symmetry = np.zeros((x1.size * times_added, ex.shape[1]))
i_sym = 0
for i, args in enumerate(zip(x1, x2, x3, ex, ey, ez)):

    def apply_symmetry(*signature):
        global x1_symmetry, x2_symmetry, x3_symmetry, ex_symmetry, ey_symmetry, ez_symmetry, i_sym
        x1_symmetry[i_sym], x2_symmetry[i_sym], x3_symmetry[i_sym], \
            ex_symmetry[i_sym, :], ey_symmetry[i_sym, :], ez_symmetry[i_sym, :] \
            = (arg*s for arg, s in zip(args, signature))
        i_sym += 1

    apply_symmetry(-1, 1, 1, -1, 1, 1)
    apply_symmetry(1, -1, 1, 1, -1, 1)
    apply_symmetry(-1, -1, 1, -1, -1, 1)

    apply_symmetry(1, 1, -1, -1, -1, 1)
    apply_symmetry(-1, 1, -1, 1, -1, 1)
    apply_symmetry(1, -1, -1, -1, 1, 1)
    apply_symmetry(-1, -1, -1, 1, 1, 1)

x1 = np.concatenate((x1, x1_symmetry))
x2 = np.concatenate((x2, x2_symmetry))
x3 = np.concatenate((x3, x3_symmetry))
ex = np.concatenate((ex, ex_symmetry), axis=0)
ey = np.concatenate((ey, ey_symmetry), axis=0)
ez = np.concatenate((ez, ez_symmetry), axis=0)

assert np.any((x1 > 0) & (x2 > 0) & (x3 > 0)) and \
      np.any((x1 > 0) & (x2 > 0) & (x3 < 0)) and \
      np.any((x1 > 0) & (x2 < 0) & (x3 > 0)) and \
      np.any((x1 > 0) & (x2 < 0) & (x3 < 0)) and \
      np.any((x1 < 0) & (x2 > 0) & (x3 > 0)) and \
      np.any((x1 < 0) & (x2 > 0) & (x3 < 0)) and \
      np.any((x1 < 0) & (x2 < 0) & (x3 > 0)) and \
      np.any((x1 < 0) & (x2 < 0) & (x3 < 0))

energy = np.sum(ex**2 + ey**2 + ez**2, axis=1)
energy_max = energy.max()
r = np.sqrt(x1**2 + x2**2 + x3**2)
plt.ion()
fig = plt.figure()
plt.plot(r, energy / energy_max, '.')
r_min = r.min()
plt.plot(r, 1 / (r/r_min)**1, '.')
plt.plot(r, 1 / (r/r_min)**2, '.')
plt.plot(r, 1 / (r/r_min)**3, '.')

plt.legend(("data", "1/r", "1/r^2", "1/r^3"))

plt.show()
input("[Enter] to continue...")

e_true = [ex, ey, ez]
print(f"{ex.shape=}")


def get_h_num(h, t):
    h_num = np.exp(-((t - t0) / gamma) ** 2) * (4 * ((t - t0) / gamma) ** 2 - 2)
    return h_num
    return scipy.interpolate.interp1d(np.linspace(t.min(), t.max(), h.size + 1 + 1),
                                      np.concatenate((np.array((0, )), h.ravel(), np.zeros((1, )))),
                                      kind="cubic")(t) * np.exp(-1/((t/0.1)**2 + 1e-16))


kwargs = {"tol": 1e-4,
          "n_points": 0,
          "error_tol": 1e-3,
          "coeff_derivative": 0,
          "verbose_every": 100,
          "plot": True,
          "scale": 1e7,
          "h_num": get_h_num,
          "find_center": False,
          "max_global_tries": 1,
          "compute_grid": False}

order = 1
shape_mom = (order + 2, order + 2, order + 2, 3)
dim_mom = 3 * order**3


def get_current_moment(moment):
    current_moment = np.zeros(shape_mom)
    current_moment[:order, :order, :order, :] = moment.reshape((order, order, order, 3))
    return current_moment


args = (order + 1, e_true, x1, x2, x3, t, get_current_moment, dim_mom)
current_moment, h, center, e_opt = inverse_problem.inverse_problem(*args, **kwargs)

if __name__ == "__main__":
    plt.ion()

    print(f"{center=}")

    inverse_problem.plot_moment(current_moment)
    inverse_problem.plot_moment(inverse_problem.get_charge_moment(current_moment))

    plt.figure()
    h -= h[0]
    plt.plot(t, h / np.max(np.abs(h)))
    plt.xlabel("Time (relative)")
    plt.ylabel("Amplitude (normalized)")
    plt.title("Current vs time")
    plt.pause(0.1)
    plt.show()
    match input("Save? [y/n] "):
        case ("y" | "Y"):
            res = pd.DataFrame(data={"t": x1, "x2": x2, "x3": x3}
                               | {f"ex_opt@t={t[i]}": e_opt[0, :, i] for i in range(ex.shape[1])}
                               | {f"ey_opt@t={t[i]}": e_opt[1, :, i] for i in range(ey.shape[1])}
                               | {f"ez_opt@t={t[i]}": e_opt[2, :, i] for i in range(ez.shape[1])}
                               | {f"ex_true@t={t[i]}": ex[:, i] for i in range(ex.shape[1])}
                               | {f"ey_true@t={t[i]}": ey[:, i] for i in range(ey.shape[1])}
                               | {f"ez_true@t={t[i]}": ez[:, i] for i in range(ez.shape[1])})
            filename = f"../../../git_ignore/GLOBALEM/opt-result-{time.asctime()}.csv"
            res.to_csv(path_or_buf=filename)
            print(f"Saved as '{filename}'.")
        case _:
            pass

