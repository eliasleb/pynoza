import numpy as np
import matplotlib.pyplot as plt
import cython
from dataclasses import dataclass

from scipy.special import erf
from scipy.special import lpmv, lpmn
from warnings import warn

from spherical_int import primes

print(primes(1000))


@dataclass
class Nu:
    s: bool
    n: int
    m: int


def smooth_step(x):
    return .5 * (erf(4 * x) + 1)


@cython.cfunc
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
        epsilon(m) * (2 * n + 1) / (4 * np.pi) * np.math.factorial(n - m) / np.math.factorial(n + m)
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
    if type(t) is not float:
        max_e, e1, e2 = np.max(np.abs(e_t)), e_t[0], e_t[-1]
        fom = max(e1/max_e, e2/max_e)
        if fom > 1e-2:
            warn(f"Time range too small, end points value {fom*100:.1f}")
    return e_t * smooth_step((a - rho)/eps_z) * smooth_step((-np.abs(z) + eps_z)/eps_z)


def m_nu_h(s: bool, max_order: int, t, x, y, z, current_density_x_, n_xi=50):
    xi = np.linspace(-1, 1, n_xi)
    dx3 = (x[1] - x[0]) * (y[1] - y[0]) * (z[1] - z[0])
    d_xi = xi[1] - xi[0]
    dt = t[1] - t[0]
    moments = np.zeros((max_order + 1, max_order + 1, t.size))
    n = np.array(range(max_order + 1))
    factorial_n = np.array([np.math.factorial(n_i) for n_i in n])[:, None, None]
    n = n[:, None, None]
    n_total = len(x) * len(y) * len(z) * len(xi)
    ind = 0
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
                )[:, :, 0, None]
                for i_xi, xi_i in enumerate(xi):
                    d_moments = 377 / 2 / factorial_n * (r / 2) ** n * (1 - xi_i ** 2) ** n * e_nu_h_\
                         * current_density_x_(
                        t + xi_i * r, x_i, y_i, z_i
                    )[None, None, :]
                    moments = moments + d_moments * dx3 * d_xi
                    ind += 1
                if ind % 3000 == 0:
                    print(f"{ind / n_total * 100:.1f}%")
    return moments


def main():
    c0 = 1
    max_order = 2
    a = 3 * c0
    eps_z = .01 * a
    n_xi = 100

    cx = lambda t_, x_, y_, z_: current_density_x(t_, x_, y_, z_, duration=1., a=a, eps_z=eps_z)
    dx = a / 60
    t = np.linspace(-12, 12, n_xi)
    x = np.arange(-1.1*a, 1.1*a, dx)
    y = np.arange(-1.1*a, 1.1*a, dx)
    z = np.linspace(-2 * eps_z, 2 * eps_z, 30)

    # jx = cx(t[:, None, None, None], x[None, :, None, None], y[None, None, :, None], z[None, None, None, :])
    # max_jx = np.max(np.abs(jx))
    # for ind, ti in enumerate(t):
    #     plt.clf()
    #     plt.subplot(1, 3, 1)
    #     plt.contourf(x, y, jx[ind, :, :, z.size//2].T, levels=np.linspace(-max_jx, max_jx, 21), cmap="jet")
    #     plt.subplot(1, 3, 2)
    #     plt.contourf(x, z, jx[ind, :, y.size//2, :].T, levels=np.linspace(-max_jx, max_jx, 21), cmap="jet")
    #     plt.subplot(1, 3, 3)
    #     plt.contourf(y, z, jx[ind, x.size//2, :, :].T, levels=np.linspace(-max_jx, max_jx, 21), cmap="jet")
    #     plt.title(f"{ti=:.1f}")
    #     plt.waitforbuttonpress()

    plt.plot(t, excitation(t))

    moments_spherical = m_nu_h(s=False, max_order=max_order, t=t, x=x, y=y, z=z, current_density_x_=cx, n_xi=n_xi)

    plt.figure()
    for n in range(max_order + 1):
        for m in range(0, n + 1):
            plt.plot(t, moments_spherical[n, m, :].T, label=f"{n=}, {m=}")

    plt.legend()
    plt.tight_layout()
    plt.show()
    # jx = cx(t, x, y, z)
    # max_jz = np.max(np.abs(jx))
    # for ind, ti in enumerate(t.squeeze()):
    #     plt.clf()
    #     plt.contourf(x.squeeze(), y.squeeze(), jx[ind, :, :, z.size//2].T,
    #                  levels=np.linspace(-max_jz, max_jz, 11), cmap="jet")
    #     plt.waitforbuttonpress()


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("TkAgg")
    main()
