import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import pandas as pd
import sympy.functions.special
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import itertools

import pynoza
import scipy.interpolate
import scipy.optimize
import re
import importlib

case_ = "fast"

match case_:
    case "fast":
        filename = "../../../git_ignore/GLOBALEM/hira_v1.txt"
    case "full":
        filename = "../../../git_ignore/GLOBALEM/hira_v4.txt"
    case _:
        filename = "../../../git_ignore/GLOBALEM/hira_v4.txt"

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
shift = 0
dt = 6.6666e-11

src_range_x = [-r_HIRA, r_HIRA]
src_range_y = [shift - r_HIRA/2, shift - r_HIRA/2 + 2*r_HIRA]
src_range_z = [0, r_HIRA]

src_center = (np.mean(src_range_x), np.mean(src_range_y), np.mean(src_range_z))
src_radius = 2*r_HIRA

if case_ == "fast":
    obs_range = [1.7*src_radius, 1.71*src_radius]
else:
    obs_range = [1.7 * src_radius, 2 * src_radius]

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

x1 = data["x"][indices_obs]
x2 = data["y"][indices_obs]
x3 = data["z"][indices_obs]
ex = data.iloc[indices_obs, 3:3*Nt+3:3]
ey = data.iloc[indices_obs, 4:3*Nt+3:3]
ez = data.iloc[indices_obs, 5:3*Nt+3:3]

assert np.all(["Ex" in name for name in ex.columns])
assert np.all(["Ey" in name for name in ey.columns])
assert np.all(["Ez" in name for name in ez.columns])

x1 = np.array(x1)
x2 = np.array(x2)
x3 = np.array(x3)

ex = np.array(ex)
ey = np.array(ey)
ez = np.array(ez)

M = 1
sol = pynoza.Solution(max_order=M,
                      wave_speed=1,)
sol.recurse()


def integrate_array(x):
    return np.cumsum(x)*dt


def derivative(x):
    return np.gradient(x, dt)


def get_all_orders(h):
    h_dict = {-1: integrate_array(h), 0: h}

    for i in range(1, M+3):
        h_dict[i] = derivative(h_dict[i-1])

    return h_dict


def get_fields(current_moment, charge_moment, h):

    h_dict = get_all_orders(h)

    c_mom = lambda a1, a2, a3: list(current_moment[a1, a2, a3])
    r_mom = lambda a1, a2, a3: list(charge_moment[a1, a2, a3])

    sol.set_moments(c_mom, r_mom)

    return sol.compute_e_field(x1,
                               x2,
                               x3,
                               t,
                               h_dict,
                               None)


e_true = [ex, ey, ez]

charge_moments = np.ones((sol.max_order+1, sol.max_order+1, sol.max_order+1, 3))
current_moments = charge_moments.copy()
Nmom = charge_moments.size
shape_mom = charge_moments.shape

h = np.sin(t).reshape(1, 1, 1, t.size)
Nh = h.size
shape_h = h.shape



def ravel_params(charge_moment, current_moments, h):
    return np.concatenate((np.ravel(charge_moment), np.ravel(current_moments), np.ravel(h)))


def unravel_params(params):
    return params[:Nmom].reshape(shape_mom), \
           params[Nmom:Nmom+Nmom].reshape(shape_mom), \
           params[2*Nmom:]


x0 = ravel_params(charge_moments, current_moments, h)


def get_error(x):

    current_moment, charge_moment, h = unravel_params(x)

    e_opt = get_fields(current_moment, charge_moment, h)

    error = 0
    for c1, c2 in zip(e_true, e_opt):
        error += np.sum((c1 - c2)**2)

    return error


get_error(x0)

# scipy.optimize.minimize(get_error, x0)
