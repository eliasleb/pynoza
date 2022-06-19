import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import inverse_problem

case_ = "full"

match case_:
    case "fast":
        filename = "../../../git_ignore/GLOBALEM/hira_v1.txt"
    case "full":
        filename = "../../../git_ignore/GLOBALEM/hira_v6.txt"
    case _:
        raise NotImplementedError()

data = pd.read_csv(filename,
                   delim_whitespace=True,
                   header=8)

names = ["x", "y", "z"]
t = 0
while len(names) < len(data.columns):
    names.append(f"Ex@{t=}")
    names.append(f"Ey@{t=}")
    names.append(f"Ez@{t=}")
    t += 1

names = names[:len(data.columns)]

data.set_axis(names, axis=1, inplace=True)
data.head()

r_HIRA = 0.7
shift = -.4
dt = 3.333333e-11

src_range_x = [-r_HIRA, r_HIRA]
src_range_y = [shift - r_HIRA/2, shift - r_HIRA/2 + 2*r_HIRA]
src_range_z = [0, r_HIRA]

src_center = (np.mean(src_range_x), np.mean(src_range_y), np.mean(src_range_z))
src_radius = 2*r_HIRA

if case_ == "fast":
    obs_range = [1.7*src_radius, 1.71*src_radius]
else:
    obs_range = [1.3 * src_radius, 20 * src_radius]

Np = data.shape[0]
Nt = 0
while not np.isnan(data.iloc[0, 3+3*Nt]):
    Nt += 1

c0 = 3e8
dt *= c0
t = np.arange(0, Nt*dt, dt)

indices_obs = list()

for i, (xi, yi, zi) in enumerate(zip(data["x"], data["y"], data["z"])):
    dist = np.sqrt((xi - src_center[0])**2 + (yi - src_center[1])**2 + (zi - src_center[2])**2)
    if obs_range[0] <= dist <= obs_range[1]:
        indices_obs.append(i)

print(f"Found {len(indices_obs)} points")

n_down_sample_t = 1
n_down_sample_x = 100

x1 = data["x"][indices_obs]
x2 = data["y"][indices_obs]
x3 = data["z"][indices_obs]
x1 = x1 - src_center[0]
x2 = x2 - src_center[1]
x3 = x3 - src_center[2]

ex = data.iloc[indices_obs, 3:3*Nt+3:3]
ey = data.iloc[indices_obs, 4:3*Nt+3:3]
ez = data.iloc[indices_obs, 5:3*Nt+3:3]
t = t[::n_down_sample_t]

assert np.all(["Ex" in name for name in ex.columns])
assert np.all(["Ey" in name for name in ey.columns])
assert np.all(["Ez" in name for name in ez.columns])

x1 = np.array(x1)[::n_down_sample_x]
x2 = np.array(x2)[::n_down_sample_x]
x3 = np.array(x3)[::n_down_sample_x]

ex = np.array(ex)[::n_down_sample_x, ::n_down_sample_t]
ey = np.array(ey)[::n_down_sample_x, ::n_down_sample_t]
ez = np.array(ez)[::n_down_sample_x, ::n_down_sample_t]

# Symmetry along yz axis
x1_sym, x2_sym, x3_sym = np.zeros(x1.shape), np.zeros(x2.shape), np.zeros(x3.shape)
ex_sym, ey_sym, ez_sym = np.zeros(ex.shape), np.zeros(ey.shape), np.zeros(ez.shape)

for i, (x1i, x2i, x3i, exi, eyi, ezi) in enumerate(zip(x1, x2, x3, ex, ey, ez)):
    ex_sym[i, :] = exi
    ey_sym[i, :] = -eyi
    ez_sym[i, :] = -ezi
    x1_sym[i] = -x1i
    x2_sym[i] = x2i
    x3_sym[i] = x3i

x1, x2, x3, ex, ey, ez = np.concatenate((x1, x1_sym)), np.concatenate((x2, x2_sym)), np.concatenate((x3, x3_sym)), \
    np.concatenate((ex, ex_sym)), np.concatenate((ey, ey_sym)), np.concatenate((ez, ez_sym))

e_true = [ex, ey, ez]
print(f"{ex.shape=}")

kwargs = {"tol": 1e-4,
          "n_points": 50,
          "error_tol": 1e-2,
          "coeff_derivative": 0}


current_moment, charge_moment, h = inverse_problem.inverse_problem(1, e_true, x1, x2, x3, t, **kwargs)

h -= h[0]
plt.plot(t, e_true[0][:, :].T, "--")
plt.plot(t, h / np.max(np.abs(h)))
plt.show()
