from pynoza import Solution
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Pool
from time import time


class Polarity(Enum):
    X = 0
    Y = 1
    Z = 2


class Side(Enum):
    Low = 0
    High = 1


@dataclass
class Wall:
    axis: Polarity
    side: Side


@dataclass
class Source:
    pos: tuple[int, ...] = (0, 0, 0)
    polarity: Polarity = Polarity.X
    amplitude: float = 1.

    def __hash__(self):
        return hash(
            self.pos + (self.polarity.value, self.amplitude, )
        )


def reflect(all_positions: set[tuple, ...], sources: set[Source, ...], size: tuple[int, int, int] = (1, 1, 1), q=0.5,
            threshold=.1):
    new_sources, amplitude = set(), None
    for src in sources:
        for dim in Polarity:
            amplitude = src.amplitude * q
            if abs(amplitude) < threshold:
                return sources
            pos_low, pos_high = list(src.pos), list(src.pos)
            pos_low[dim.value] *= -1
            pos_high[dim.value] = 2 * size[dim.value] - src.pos[dim.value]
            if src.polarity != dim:
                amplitude *= -1
            pos_low, pos_high = tuple(pos_low), tuple(pos_high)
            if pos_low not in all_positions:
                new_sources.add(
                    Source(pos_low, polarity=src.polarity, amplitude=amplitude)
                )
                all_positions.add(pos_low)
            if pos_high not in all_positions:
                new_sources.add(
                    Source(pos_high, polarity=src.polarity, amplitude=amplitude),
                )
                all_positions.add(pos_high)
    print(len(all_positions), abs(amplitude))
    sources.update(
        reflect(
            all_positions,
            new_sources,
            size=size, q=q, threshold=threshold
        )
    )
    return sources


def get_e_field(t, x1, x2, x3, r, c, h):
    return np.array([0., -1.26e-06, 0])[:, None] * (h[1](t - r / c) / r) / (4 * np.pi) \
        + np.array([0., 0., 1.13e+17])[:, None] * ((3. * h[-1](t - r / c) * x2 * x3 / r**5) + (
                    1.00e-11 * h[0](t - r / c) * x2 * x3 / r**4) + (
                            1.11e-23 * h[1](t - r / c) * x2 * x3 / r**3)) / (4 * np.pi) \
        + 1 / 2. * np.array([0., 2.26e+17, 0.])[:, None] * (+(-1. * h[-1](t - r / c) / r**3) + (
                    3. * h[-1](t - r / c) * x2**2 / r**5) + (1.00e-11 * h[0](
                t - r / c) * x2**2 / r**4) + (-3.33e-12 * h[0](t - r / c) / r**2) + (1.11e-23 * h[1](t - r / c)
                                                                                 * x2**2 / r**3)) / (4 * np.pi) \
        + np.array([1.13e+17, 0., 0.])[:, None] * ((3. * h[-1](t - r / c) * x1 * x2 / r**5) + (
                        1.00e-11 * h[0](t - r / c) * x1 * x2 / r**4) + (
                                1.11e-23 * h[1](t - r / c) * x1 * x2 / r**3)) / (4 * np.pi)


def sum_sources(sources, solution, x1, x2, x3, t, x):
    e_field = 0.
    for ind, source in enumerate(sources):
        if ind % 500 == 0:
            print(f"{ind/len(sources)*100:.1f}")
        e_field = solution.compute_e_field(
            x1 - source.pos[0], x2 - source.pos[1], x3 - source.pos[2], t, source.amplitude * x, None,
            compute_grid=False
        ) + e_field
    return e_field


def main():

    d_com = np.loadtxt(
        "../../../git_ignore/image theory/validation_comsol.txt",
        skiprows=5
    )
    t_com, v_com = d_com[:, 0], d_com[:, 1]

    solution = Solution(max_order=2, wave_speed=3e11)
    solution.set_moments(
        current_moment=lambda a1, a2, a3: [0., 1., 0.] if (a1, a2, a3) == (0, 0, 0) else [0., 0., 0.]
    )
    solution.recurse()

    size_x, size_y, size_z = 1356, 906, 814
    x1, x2, x3 = np.array((size_x - 226, )), np.array((size_y, )), np.array((296, ))

    data = np.genfromtxt(
        "../../../project/experimental/emtr-experimental/data/oscilloscope/C3--Trace--00528.txt",
        delimiter=",",
        skip_header=5
    )
    t, h = data[:, 0], data[:, 1]
    dt = t[1] - t[0]
    fs = 1 / dt
    fc, bw = 2350e6, 100e6
    sos = scipy.signal.butter(
        2, (fc - bw, fc + bw),
        btype="bandpass",
        analog=False,
        output="sos",
        fs=fs
    )

    f = np.linspace(0, fs, t.size)
    h_fd = np.fft.fft(h)

    h = scipy.signal.sosfilt(sos, h)
    h_f_fd = np.fft.fft(h)
    h = h / np.max(h)
    t0 = 60e-9
    x = np.exp(-.5 * ((t - t0) * bw)**2) * np.cos(2 * np.pi * fc * (t - t0))
    x[t < t0/2] = 0.
    x_fd = np.fft.fft(x)

    plt.plot(f, np.abs(h_fd)/np.max(np.abs(h_fd)))
    plt.plot(f, np.abs(h_f_fd)/np.max(np.abs(h_f_fd)))
    plt.plot(f, np.abs(x_fd)/np.max(np.abs(x_fd)))
    plt.xlim(2.2e9, 2.5e9)

    src = Source(pos=(466, 0, 307), polarity=Polarity.Y)
    q = 1 - (2 * 8.854e-12 * 2 * np.pi * fc / 3.5e7)**.5
    print(f"{q=}")
    sources = reflect(
        {src.pos, },
        {src, },
        size=(size_x, size_y, size_z),
        threshold=q**(350/3.33),  # .0001
        q=q
    )
    plot_points = False
    if plot_points:
        max_line = 10
        plt.hlines([size_y * ind for ind in range(-max_line, max_line)], -max_line * size_x, max_line * size_x)
        plt.vlines([size_x * ind for ind in range(-max_line, max_line)], -max_line * size_y, max_line * size_y)

        for ind, src in enumerate(sources):
            if 306 < src.pos[2] < 308:
                print(id(src), src)
                plt.plot(src.pos[0], src.pos[1], ".", color=plt.colormaps["jet"](abs(src.amplitude)))
                plt.text(src.pos[0], src.pos[1], f"{src.amplitude:.3f}")
        plt.show()

    t_min, t_max = .34e-7, 1.2e-6
    keep = (t > t_min) & (t < t_max)
    n_downsample = 1

    t, x, h = t[keep], x[keep], h[keep]
    t, x, h = t[::n_downsample], x[::n_downsample], h[::n_downsample]

    t = t - t[0]

    start = time()
    use_parallel = True
    if not use_parallel:
        e_field = 0.
        for ind, source in enumerate(sources):
            if ind % 500 == 0:
                print(f"{ind/len(sources)*100:.1f}", source.pos, source.amplitude)
            e_field = solution.compute_e_field(
                x1 - source.pos[0], x2 - source.pos[0], x3 - source.pos[0], t, source.amplitude * x, None,
                compute_grid=False
            ) + e_field
    else:
        segment_size = len(sources) // 8
        n_processes, arguments, sources = 0, [], list(sources)
        for shift in range(0, len(sources), segment_size):
            arguments.append(
                (sources[shift:shift + segment_size], solution, x1, x2, x3, t, x)
            )
            n_processes += 1

        print(f"Using {n_processes=}")
        with Pool(n_processes) as p:
            result = p.starmap(sum_sources, arguments)
        e_field = sum(result)

    stop = time()

    print(f"It took {stop - start} s")

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t, h)

    plt.subplot(2, 1, 2)
    # plt.plot(t, x)
    # plt.plot(t_com, v_com/np.max(v_com), "k--")
    plt.plot(t, e_field[1, :, :].T)
    # plt.xlim(3e-8, 10e-8)
    plt.show()


if __name__ == "__main__":
    matplotlib.use("TkAgg")
    main()
