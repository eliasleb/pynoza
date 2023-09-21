from pynoza import Solution
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from dataclasses import dataclass
from enum import Enum


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


def main():

    solutions = []
    solution = Solution(max_order=2, wave_speed=3e11)
    solution.set_moments(
        current_moment=lambda a1, a2, a3: [0., 1., 0.] if (a1, a2, a3) == (0, 0, 0) else [0., 0., 0.]
    )
    solution.recurse()
    solutions.append(solution)

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
    fc, bw = 2.35e9, 40e6
    sos = scipy.signal.butter(
        2, (fc - bw, fc + bw),
        btype="bandpass",
        analog=False,
        output="sos",
        fs=fs
    )

    # f = np.linspace(0, fs, t.size)
    # h_fd = np.fft.fft(h)

    h = scipy.signal.sosfilt(sos, h)
    # h_f_fd = np.fft.fft(h)
    h = h / np.max(h)
    x = scipy.signal.gausspulse(
        t - 60e-9, fc, bw/fc
    )
    # x_fd = np.fft.fft(x)

    # plt.plot(f, np.abs(h_fd)/np.max(np.abs(h_fd)))
    # plt.plot(f, np.abs(h_f_fd)/np.max(np.abs(h_f_fd)))
    # plt.plot(f, np.abs(x_fd)/np.max(np.abs(x_fd)))
    # plt.xlim(2.2e9, 2.5e9)
    # plt.show()

    src = Source(pos=(466, 0, 307), polarity=Polarity.Y)
    sources = reflect(
        {src.pos, },
        {src, },
        size=(size_x, size_y, size_z),
        threshold=.5,
        q=(1/2) ** (1/(3e8*2.5e-7))
    )
    if False:
        plt.hlines([0, size_y], 0, size_x)
        plt.vlines([0, size_x], 0, size_y)

        for ind, src in enumerate(sources):
            if 306 < src.pos[2] < 308:
                print(id(src), src)
                plt.plot(src.pos[0], src.pos[1], ".")
                plt.text(src.pos[0], src.pos[1], f"{str(id(src))[-3:]}")
                plt.waitforbuttonpress()

    t_min, t_max = -.01e-7, .2e-6
    keep = (t > t_min) & (t < t_max)
    n_downsample = 3

    t, x, h = t[keep], x[keep], h[keep]
    t, x, h = t[::n_downsample], x[::n_downsample], h[::n_downsample]
    print(len(sources))
    e_field = 0.
    for ind, source in enumerate(sources):
        if ind % 500 == 0:
            print(f"{ind/len(sources)*100:.1f}", source.pos, source.amplitude)
        e_field = solution.compute_e_field(
            x1 - source.pos[0], x2 - source.pos[0], x3 - source.pos[0], t, source.amplitude * x, None,
            compute_grid=False
        ) + e_field

    plt.figure()
    plt.plot(t, h / np.max(h))
    # plt.plot(t, x)
    plt.plot(t, e_field[1, 0, :] / np.max(e_field), "k--")
    plt.show()


if __name__ == "__main__":
    matplotlib.use("TkAgg")
    main()
