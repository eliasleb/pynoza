import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import inverse_problem
import itertools
import scipy.interpolate


filename = "../../../git_ignore/GLOBALEM/dipole_at_obs_v4.txt"
data = pd.read_csv(filename,
                   delim_whitespace=True,
                   header=8)
names = ["Time", ]
with open(filename) as f:
    for i, line in enumerate(f):
        if i == 7:
            names.extend(line.split(" Electric field, ")[1:])
data.set_axis(names, axis=1, inplace=True)

r_HIRA = 0.7
shift = -0.4

src_range_x = [-r_HIRA, r_HIRA]
src_range_y = [shift - r_HIRA/2, shift - r_HIRA/2 + 2*r_HIRA]
src_range_z = [0, r_HIRA]

src_center = (np.mean(src_range_x), np.mean(src_range_y), np.mean(src_range_z))
src_radius = 3*r_HIRA

c0 = 3e8
t = np.array(data["Time"])*c0
d_obs = 0.15#4*r_HIRA
# x1 = np.array([-d_obs, 0, ]).reshape((2, 1, 1))
# x2 = np.array([-d_obs, 0, d_obs, ]).reshape((1, 3, 1))
# x3 = np.array([-d_obs, 0, d_obs, ]).reshape((1, 1, 3))
x1 = np.array([0, ]).reshape((1, 1, 1))
x2 = np.array([0, ]).reshape((1, 1, 1))
x3 = np.array([-d_obs, d_obs, ]).reshape((1, 1, 2))

x2 = x2 + 0.5

ex = np.zeros((x1.size, x2.size, x3.size, t.size))
ey, ez = ex.copy(), ex.copy()


def get_str_descr(x, name="2*r_HIRA"):
    if x > 0:
        return name
    elif x < 0:
        return "-" + name
    else:
        return "0"

num_col = 0
for i, j, k in itertools.product(range(x1.size), range(x2.size), range(x3.size)):
    x1i, x2i, x3i = x1[i, 0, 0], x2[0, j, 0], x3[0, 0, k]
    str_descr = f"({get_str_descr(x1i)}, {get_str_descr(x2i)}, {get_str_descr(x3i)})"
    if x1i**2 + x2i**2 + x3i**2 < 1e-6: continue
    num_components = 0
    for i_col, col in enumerate(data.columns):
        if i_col == 0: continue
        if str_descr in col:
            if "x component" in col:
                ex[i, j, k, :] = data[col].values
                num_components += 1
            elif "y component" in col:
                ey[i, j, k, :] = data[col].values
                num_components += 1
            elif "z component" in col:
                ez[i, j, k, :] = data[col].values
                num_components += 1
            else:
                raise RuntimeError(f"Unknown column: {col}")
    if num_components != 3:
        raise RuntimeError(f"Missing components for {str_descr} (has columns {data.columns})")
    num_col += 1


e_true = [ex, ey, ez]
print(f"{ex.shape=}")


def get_h_num(h, t):
    h[-h.size // 3:] = 0
    h[0] = 0
    return scipy.interpolate.interp1d(np.linspace(t.min(), t.max(), h.size),
                                      h,
                                      kind="cubic")(t) * np.exp(-1/((t/0.1)**2 + 1e-16))

kwargs = {"tol": 1e-6,
          "n_points": 40,
          "error_tol": 1e-3,
          "coeff_derivative": 0,
          "verbose_every": 100,
          "plot": True,
          "scale": 1e6,
          "h_num": get_h_num,
          "find_center": False}

order = 1
args = (order, e_true, x1, x2, x3, t,)
current_moment, charge_moment, h, center = inverse_problem.inverse_problem(*args, **kwargs)

print(f"{current_moment=}")
print(f"{charge_moment=}")
print(f"{center=}")

if order >= 1:
    def plot_moment(moment):
        plt.figure()
        m = np.max(np.abs(moment))
        kwargs = {"cmap": "RdBu", "vmin": -m, "vmax": m}
        for comp, iz in itertools.product(range(3), range(order+1)):
            plt.subplot(3, order+1, 2*comp + iz + 1)
            plt.imshow(current_moment[:, :, iz, comp].T, **kwargs)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(f"{comp=}")

        plt.tight_layout()

    plot_moment(current_moment)
    plot_moment(charge_moment)

plt.pause(0.001)
input("Press [enter] to exit.")
plt.close("all")


