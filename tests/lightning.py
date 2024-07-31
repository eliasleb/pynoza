import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from itertools import product
import inverse_problem
from scipy.interpolate import interp1d


def read_data(filename):
    raw_data = loadmat(filename)
    t, e_r, e_z, h_phi = raw_data["time"], raw_data["er"], raw_data["ez"], raw_data["hphi"]
    return t.squeeze(), e_r.squeeze(), e_z.squeeze(), h_phi.squeeze()


def read_all_data(window_us=1000):
    # r <> x, y <> not used, z <> z
    # r_km = np.array((1., 5., 10.))
    # z_km = np.array((0., 2., 4.))
    r_km = np.array((5., 10., ))
    z_km = np.array((0., ))

    t, e_field, h_field, n_t = None, None, None, None
    ind_x = 0
    x, y, z = [], [], []
    for r in r_km:
        for zi in z_km:
            _t, e_r, e_z, h_phi = read_data(
                f"data/lightning_data/20240730/EMF_T={window_us}_r={int(r)}_z={int(zi)}.mat"
            )
            if e_field is None:
                e_field = np.zeros((3, r_km.size * z_km.size, _t.size))
                h_field = e_field.copy()
                t = _t
                n_t = t.size
            e_field[:, ind_x, :] = np.array((
                e_r[:n_t], np.zeros((n_t, )), e_z[:n_t]
            ))
            h_field[:, ind_x, :] = np.array((
                np.zeros((n_t, )), h_phi[:n_t], np.zeros((n_t, ))
            ))
            x.append(r)
            y.append(0)
            z.append(zi)
            ind_x += 1
    x, y, z = np.array(x) * 1e3, np.array(y), np.array(z) * 1e3
    return x, y, z, t, e_field, h_field


def lightning_inverse_problem(**kwargs):
    max_order = kwargs.pop("max_order", 0)
    plot = kwargs.pop("plot", False)
    verbose_every = kwargs.pop("verbose_every", 100)
    scale = kwargs.pop("scale", 1e0)
    tol = kwargs.pop("tol", 1e-4)
    n_points = kwargs.pop("n_points", 50)
    find_center = kwargs.pop("find_center", False)

    x, y, z, t, e_field, h_field = read_all_data()
    e_field /= 1e2
    c0 = 3e8
    t *= c0
    # x = x / c0
    # y = y / c0
    # z = z / c0

    moments_shape = (3, ) + (max_order + 3, ) * 3
    dim_moment = 1 * (max_order + 1)

    def get_current_moment(moment):
        _current_moment = np.zeros(moments_shape)
        ind = 0
        for az in range(max_order + 1):
            _current_moment[2, 0, 0, az] = moment[ind] / 10**az
            ind += 1

        assert ind == moment.size
        return _current_moment

    def get_h_num(h_, t_):
        h_interpolated = interp1d(
            np.linspace(np.min(t_), np.max(t_), 1 + h_.size),
            np.concatenate((np.array((0,)), h_.ravel())),
            kind="cubic"
        )(t_)
        return h_interpolated

    current_moment, h, center, e_opt = inverse_problem.inverse_problem(
        order=max_order + 2,
        e_true=e_field,
        x1=x,
        x2=y,
        x3=z,
        t=t,
        _t_sym=None,
        current_moment_callable=get_current_moment,
        dim_moment=dim_moment,
        h_num=get_h_num,
        plot=plot,
        verbose_every=verbose_every,
        scale=scale,   # shift, scale: 0, 14 -- -1 22
        tol=tol,
        n_points=n_points,
        find_center=find_center,
        shift=0,
        rescale_at_points=True
    )

    if plot:
        plt.ion()

        print(f"{center=}")

        inverse_problem.plot_moment(current_moment)
        plt.figure()
        max_h = np.max(np.abs(h))
        if max_h > 0:
            plt.plot(t, h / max_h)
        plt.xlabel("Time (relative)")
        plt.ylabel("Amplitude (normalized)")
        plt.title("Current vs time")

        print(np.max(np.abs(current_moment)))

        plt.pause(0.1)
        plt.show(block=True)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("TkAgg")
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--max_order", type=int, required=True)
    parser.add_argument("--plot", type=bool, default=False)
    parsed = parser.parse_args()
    lightning_inverse_problem(**vars(parsed))
