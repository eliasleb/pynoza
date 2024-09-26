import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from dataclasses import dataclass
import math
from scipy.special import erf
from scipy.special import lpmv, lpmn
import time
import itertools
from alive_progress import alive_bar


@dataclass
class Nu:
    s: bool
    n: int
    m: int


@dataclass
class Parameters:
    a: float  # 3.
    max_order: int  # 2
    eps_z: float  # .03
    n_rho: int  # 4
    n_z: int  # 4
    n_xi: int  # 1000


def smooth_step(x):
    return .5 * (erf(4 * x) + 1)


def epsilon(m: int):
    return 1 if m == 0 else 2


def phi_nu_alpha(nu: Nu, theta, phi):
    if nu.s is False:
        f = np.cos
    else:
        f = np.sin
    return ynm(nu.n, nu.m) * lpmv(nu.n, nu.m, np.cos(theta)) * f(nu.m * phi)


def ynm(n: int, m: int):
    return (
        epsilon(m) * (2 * n + 1) / (4 * np.pi) * math.factorial(n - m) / math.factorial(n + m)
           )**.5


def e_theta(theta, phi):
    return np.array([
        np.cos(theta) * np.cos(phi),
        np.cos(theta) * np.sin(phi),
        -np.sin(theta)
    ])


def e_phi(phi):
    return np.array([np.sin(phi), np.cos(phi), 0])


def e_nu_h(s: bool, max_order: int, theta: float, phi: float):
    f1 = np.cos if not s else np.sin
    f2 = np.sin if not s else lambda x_: -np.cos(x_)

    pmn, dpmn = lpmn(max_order, max_order, np.cos(theta))
    ret = np.zeros(pmn.shape + (3, ), dtype=float)
    for ind_m, m in enumerate(range(max_order + 1)):
        for ind_n, n in enumerate(range(max_order + 1)):
            if n == 0 or m > n:
                continue
            ret[ind_n, ind_m] = ynm(n, m) / (n * (n + 1))**.5 * (
                - e_phi(phi)[None, None, :] * (-1)**m * dpmn[ind_m, ind_n, None] * np.sin(theta)
                * f1(m * phi)
                + e_theta(theta, phi)[None, None, :] * m / np.sin(theta)
                * (-1)**m * pmn[ind_m, ind_n, None] * f2(m * phi)
            )
    return ret


def excitation(t, big_t=1.):
    gamma = (12 / 7)**.5 * big_t
    return (3 * gamma * (np.pi / 2))**(-1/2) * np.exp(-(t/gamma)**2) * (4 * (t/gamma)**2 - 2)


def current_density_x(t, x, y, z, duration=1., a=1., eps_z=.1):
    rho = np.sqrt(x**2 + y**2)
    e_t = excitation(t, duration)
    # if type(t) is not float:
    #     max_e, e1, e2 = np.max(np.abs(e_t)), e_t[0], e_t[-1]
    #     fom = max(e1/max_e, e2/max_e)
    #     if fom > 1e-2:
    #         warn(f"Time range too small, end points value {fom*100:.1f}")
    return e_t * smooth_step((a - rho)/eps_z) * smooth_step((-np.abs(z) + eps_z)/eps_z)


def m_nu_h(s: bool, max_order: int, t, x, y, z, current_density_x_, n_xi=50):
    xi = np.linspace(-1, 1, n_xi)
    dx3 = (x[1] - x[0]) * (y[1] - y[0]) * (z[1] - z[0])
    d_xi = xi[1] - xi[0]
    dt = t[1] - t[0]
    moments = np.zeros((max_order + 1, max_order + 1, t.size))
    n = np.array(range(max_order + 1))
    factorial_n = np.array([math.factorial(n_i) for n_i in n])[:, None, None, None]
    n = n[:, None, None, None]
    n_total = len(x) * len(y) * len(z)
    with alive_bar(n_total, bar="bubbles", spinner="notes") as bar:
        for i_x, x_i in enumerate(x):
            for i_y, y_i in enumerate(y):
                rho = (x_i ** 2 + y_i ** 2) ** .5
                phi = np.arctan2(y_i, x_i)
                for i_z, z_i in enumerate(z):
                    r = (x_i ** 2 + y_i ** 2 + z_i ** 2) ** .5
                    theta = np.arctan2(z_i, rho)
                    e_nu_h_ = e_nu_h(
                        s=s,
                        max_order=max_order,
                        theta=theta,
                        phi=phi
                    )[:, :, 0, None, None]
                    # for i_xi, xi_i in enumerate(xi):
                    d_moments = 377 / 2 / factorial_n * (r / 2) ** n * (1 - xi[None, None, None, :] ** 2) ** n \
                         * e_nu_h_ * current_density_x_(
                        t[:, None] + xi[None, :] * r, x_i, y_i, z_i
                    )[None, None, ...]
                    moments = moments + np.sum(d_moments, axis=-1) * dx3 * d_xi
                    bar()
    return moments


def cartesian_moment(alpha: tuple[int, int, int], t, x, y, z, current_density_x_):
    moment = np.zeros(t.shape)
    dx3 = (x[1] - x[0]) * (y[1] - y[0]) * (z[1] - z[0])
    n_total = x.size * y.size * z.size
    with alive_bar(n_total, bar="bubbles", spinner="notes") as bar:
        for i_x, x_i in enumerate(x):
            for i_y, y_i in enumerate(y):
                for i_z, z_i in enumerate(z):
                    moment = moment + dx3 * current_density_x_(t, x_i, y_i, z_i) * \
                        x_i**alpha[0] * y_i**alpha[1] * z_i**alpha[2]
                    bar()
    return moment


def diff_moments(moments, dt):
    max_order = moments.shape[0] - 1
    moments = np.diff(moments, axis=-1, prepend=np.zeros((max_order + 1, max_order + 1, 1))) / dt
    for ni in range(1, max_order + 1):
        moments[ni:, :, :] = np.diff(
            moments[ni:, :, :], axis=-1, prepend=np.zeros((max_order + 1 - ni, max_order + 1, 1))
        ) / dt
    return moments


def get_even(n: int):
    if n % 2 == 0:
        return n
    return n + 1


def test_moments():
    import pickle
    nx = 10

    p = Parameters(
        a=3.,
        eps_z=.03,
        n_rho=nx,
        n_z=nx,
        n_xi=100,
        max_order=2
    )

    cx = lambda t_, x_, y_, z_: current_density_x(t_, x_, y_, z_, duration=1., a=p.a, eps_z=p.eps_z)
    t = np.linspace(-12, 12, p.n_xi)
    x = np.linspace(-1.1*p.a, 1.1*p.a, p.n_rho)
    y = np.linspace(-1.1*p.a, 1.1*p.a, p.n_rho)
    z = np.linspace(-2 * p.eps_z, 2 * p.eps_z, p.n_z)

    try:
        with open("dump.pickle", "rb") as fd:
            moments_spherical, moment_cartesian, p_dumped = pickle.load(fd)
            if p_dumped != p:
                raise FileNotFoundError
    except FileNotFoundError:
        moments_spherical = m_nu_h(s=True, max_order=p.max_order, t=t, x=x, y=y, z=z, current_density_x_=cx,
                                   n_xi=p.n_xi)
        moment_cartesian = cartesian_moment((1, 0, 1), t, x, y, z, cx)
        with open("dump.pickle", "wb") as fd:
            pickle.dump((moments_spherical, moment_cartesian, p), fd)

    moments_spherical = diff_moments(moments_spherical, t[1] - t[0])

    plt.figure()
    moment_spherical = moments_spherical[2, 1, :]
    plt.plot(t, moment_spherical/np.max(np.abs(moment_spherical)), label=f"2, 1")
    plt.plot(t, moment_cartesian/np.max(np.abs(moment_cartesian)), label="(1, 1, 0)")

    plt.legend()
    plt.tight_layout()
    plt.show()


def time_function(f: callable, n_times: int = 1):
    times = np.zeros((n_times, ))
    for i in range(n_times):
        start_time = time.process_time()
        f()
        stop_time = time.process_time()
        times[i] = stop_time - start_time
    return np.mean(times)


def permutator(iterator):
    import random
    random.seed(0)
    items = list(iterator)
    shuffler = list(range(len(items)))
    random.shuffle(shuffler)
    return [items[ind] for ind in shuffler], shuffler


def time_computation(n_x, _n_xi):
    p = Parameters(
        a=3.,
        eps_z=.03,
        n_rho=n_x,
        n_z=n_x,
        n_xi=2 * n_x,
        max_order=2
    )

    cx = lambda t_, x_, y_, z_: current_density_x(t_, x_, y_, z_, duration=1., a=p.a, eps_z=p.eps_z)
    t = np.linspace(-12, 12, p.n_xi)
    x = np.linspace(-1.1*p.a, 1.1*p.a, p.n_rho)
    y = np.linspace(-1.1*p.a, 1.1*p.a, p.n_rho)
    z = np.linspace(-2 * p.eps_z, 2 * p.eps_z, p.n_z)

    compute_spherical = lambda: m_nu_h(
        s=True, max_order=p.max_order, t=t, x=x, y=y, z=z, current_density_x_=cx, n_xi=p.n_xi
    )
    compute_cartesian = lambda: cartesian_moment((1, 0, 1), t, x, y, z, cx)
    time_spherical = time_function(compute_spherical)
    time_cartesian = time_function(compute_cartesian)
    return time_spherical, time_cartesian


def main():
    import pickle

    n_t_s = np.array((100, ))
    n_x_s = [get_even(int(ni)) for ni in np.logspace(1.3, 2.2, 30)]
    n_times = 3
    shuffled, shuffler = permutator(itertools.product(n_x_s, n_t_s))

    try:
        with open(f"time_result.pickle", "rb") as fd:
            n_t_s, n_x_s, shuffled, shuffler, parallel_result = pickle.load(fd)

    except FileNotFoundError:
        with multiprocessing.Pool(8) as pool:
            parallel_result = pool.starmap(time_computation, [
                (n_x, n_t) for n_x, n_t in shuffled
            ] * n_times)
        # for i_t, n_t in enumerate(n_t_s):
        #     for i_x, n_x in enumerate(n_x_s):
        #         sph_times[i_t, i_x], car_times[i_t, i_x] = time_computation(n_x, n_t)
        with open(f"time_result.pickle", "wb") as fd:
            pickle.dump((n_t_s, n_x_s, shuffled, shuffler, parallel_result), fd)

    n_x_s = np.array(n_x_s)

    sph_times_full = np.zeros((n_t_s.size, n_x_s.size, n_times))
    car_times_full = sph_times_full.copy()

    for ind, ((n_x, n_t), (t_sph, t_car)) in enumerate(zip(
        shuffled * n_times, parallel_result
    )):
        i_x, i_t = np.where(n_x_s == n_x)[0][0], np.where(n_t_s == n_t)[0][0]
        sph_times_full[i_t, i_x, ind // (len(shuffled))] = t_sph
        car_times_full[i_t, i_x, ind // (len(shuffled))] = t_car

    sph_times = np.mean(sph_times_full, axis=-1)
    car_times = np.mean(car_times_full, axis=-1)
    n_thresh = 26
    ind = n_x_s >= n_thresh

    x = np.log10(n_x_s)
    y1 = np.log10(sph_times[-1, :])
    y2 = np.log10(car_times[-1, :])

    poly1, poly2 = np.polyfit(x[ind], y1[ind], 1), np.polyfit(x[ind], y2[ind], 1)
    y1p, y2p = np.polyval(poly1, x), np.polyval(poly2, x)
    print(f"{poly1=}, {poly2=}")

    plt.figure(figsize=(5, 3))
    plt.scatter(x, y1, label="spherical", facecolors='k', marker="+", s=30)
    plt.plot(x[ind], y1p[ind], "k:", label="spherical, fit")
    plt.scatter(x, y2, label="Cartesian", facecolors='none', edgecolors='r', s=15)
    plt.plot(x[ind], y2p[ind], "r--", label="Cartesian, fit",)

    # plt.plot(n_x_s, np.log10((n_x_s/n_x_s[-1])**4 * sph_times[-1, -1]/sph_times[-1, 1]), "r--")
    # plt.plot(n_x_s, np.log10((n_x_s/n_x_s[-1])**3 * car_times[-1, -1]/car_times[-1, 1]), "k--")

    plt.legend()

    plt.xlim(np.log10(n_thresh), 2.2)
    # plt.ylim(0, 3.5)

    plt.xlabel("Log10 of number of samples N")
    plt.ylabel("Log10(s)")

    plt.title("Moments computation time")

    plt.tight_layout()

    plt.savefig("data/moments_time.pdf")

    plt.show()


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("TkAgg")
    main()
