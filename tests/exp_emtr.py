from pynoza import Solution
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import itertools
from dataclasses import dataclass
from enum import Enum, auto


def dipole_field(solution, r0, x1, x2, x3, t, h):
    return solution.compute_e_field(
        x1 - r0[0], x2 - r0[1], x3 - r0[0],
        t, h, None
    )


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
    pos: tuple[float, ...] = (0., 0., 0.)
    polarity: Polarity = Polarity.X
    amplitude: float = 1.
    last_reflection: Wall = None
    reflected_src: int = None


def reflect(sources: list[Source, ...], size=(1., 1., 1.), q=0.5, threshold=.1):
    new_sources = []
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
            if src.last_reflection != Wall(side=Side.Low, axis=dim):
                new_sources.append(
                    Source(tuple(pos_low), polarity=src.polarity, amplitude=amplitude,
                           last_reflection=Wall(side=Side.Low, axis=dim), reflected_src=id(src))
                )
            if src.last_reflection != Wall(side=Side.High, axis=dim):
                new_sources.append(
                    Source(tuple(pos_high), polarity=src.polarity, amplitude=amplitude,
                           last_reflection=Wall(side=Side.High, axis=dim), reflected_src=id(src)),
                )

    return sources + reflect(
        new_sources,
        size=size, q=q, threshold=threshold
    )


def main():

    solutions = []
    for polarity in Polarity:
        solution = Solution(max_order=2, wave_speed=3e11)
        moment = [0., 0., 0.]
        moment[polarity.value] = 1.
        solution.set_moments(
            current_moment=lambda a1, a2, a3: moment if (a1, a2, a3) == (0, 0, 0) else [0., 0., 0.]
        )
        solution.recurse()
        solutions.append(solution)

    size_x, size_y, size_z = 1356., 906., 814.
    x1, x2, x3 = np.array((size_x - 226, )), np.array((size_y, )), np.array((296, ))

    data = np.genfromtxt(
        "../../../project/experimental/emtr-experimental/data/oscilloscope/C3--Trace--00528.txt",
        delimiter=",",
        skip_header=5
    )
    t, h = data[:, 0], data[:, 1]
    dt = t[1] - t[0]
    fs = 1 / dt
    fc, bw = 2.35e9, 100e6
    sos = scipy.signal.butter(
        2, (fc - bw, fc + bw),
        btype="bandpass",
        analog=False,
        output="sos",
        fs=fs
    )

    h = scipy.signal.sosfilt(sos, h)
    h = h / np.max(h)
    x = scipy.signal.gausspulse(
        t, fc, bw/fc
    )
    src = Source(pos=(466, 30, 307))
    sources = reflect(
        [src, ],
        size=(size_x, size_y, size_z),
        threshold=0.5,
        q=0.9
    )
    #plt.hlines([0, size_y], 0, size_x)
    #plt.vlines([0, size_x], 0, size_y)
    #
    #for ind, src in enumerate(sources):
    #    if 306 < src.pos[2] < 308:
    #        print(id(src), src)
    #        plt.plot(src.pos[0], src.pos[1], ".")
    #        plt.text(src.pos[0], src.pos[1], f"{str(id(src))[-3:]}")
    #        plt.waitforbuttonpress()

    n_downsample = 10

    e_field = None
    t, x, h = t[::n_downsample], x[::n_downsample], h[::n_downsample]

    for ind, source in enumerate(sources):
        print(f"{ind/len(sources)*100:.1f}", source.pos, source.amplitude)
        contribution = dipole_field(
            solutions[source.polarity.value],
            source.pos,
            x1, x2, x3, t, source.amplitude * x
        )
        if e_field is None:
            e_field = contribution
        else:
            e_field = e_field + contribution

    plt.figure()
    # plt.plot(t, h)
    # plt.plot(t, x)
    plt.plot(t, e_field[1, 0, 0, 0, :])
    plt.show()


if __name__ == "__main__":
    matplotlib.use("TkAgg")
    main()
