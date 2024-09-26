import pynoza
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools

matplotlib.use("TkAgg")


def current_integral(x0, sigma, power):
    return 1 / (power + 1) * ((x0 + sigma / 2) ** (power + 1) - (x0 - sigma / 2) ** (power + 1))


def current_moment_rectangle(ax, ay, _az, x, y, width, height):
    x += width / 2
    y += height / 2
    return current_integral(x, width, ax) * current_integral(y, height, ay)


def demo_logo():
    gamma_si = 4e-9
    gamma = 1.
    c0 = 3e8 * gamma_si

    order = 8

    t = np.linspace(0, 10 * gamma, 200)
    hp = (3 * gamma_si * np.sqrt(np.pi / 2)) ** -.5 \
         * np.exp(-((t-3*gamma)/gamma)**2) * (4*((t-3*gamma)/gamma)**2-2)

    t_g = (7 / 12)**.5
    wavelength = c0 * t_g

    # Logo dimensions
    a = wavelength / 2
    length = 2 * a

    # Exact logo rectangles
    length_logo = 171
    aspect_ratio = 50 / length_logo
    x_coordinates = [0, 0, 10, 0, 0, 45, 45, 45, 45, 91, 91, 101, 91, 136, 136]
    y_coordinates = [0, 10, 20, 30, 40, 0, 20, 30, 40, 0, 30, 20, 40, 10, 0]
    widths = [35, 10, 23, 10, 35, 10, 23, 10, 23, 10, 10, 23, 35, 10, 35]
    heights = [10, 10, 10, 10, 10, 20, 10, 10, 10, 20, 10, 10, 10, 40, 10]
    shift_x = a
    shift_y = a * aspect_ratio

    # Approximation of the curvy part of the `P'
    pixel_width = 1
    pixel_height = 1
    x_coordinate_candidates = np.arange(0, length_logo, pixel_width)
    y_coordinate_candidates = np.arange(0, length_logo, pixel_height)
    for x, y in itertools.product(x_coordinate_candidates, y_coordinate_candidates):
        if 15**2 >= (x - 60)**2 + (y - 35)**2 >= 5**2 and x > 60:
            x_coordinates.append(x)
            y_coordinates.append(y)
            widths.append(pixel_width)
            heights.append(pixel_height)

    def current_moment(ax, ay, az):
        moment = 0
        if az == 0:
            for x_i, y_i, w_i, h_i in zip(x_coordinates, y_coordinates, widths, heights):
                moment += current_moment_rectangle(
                    ax, ay, az,
                    x_i / length_logo * length - shift_x,
                    y_i / length_logo * length - shift_y,
                    w_i / length_logo * length,
                    h_i / length_logo * length
                ) / gamma_si
        return [moment, 0, 0]

    x = np.array([0, ])
    y = np.array([0, ])
    z = np.array([2, 3, ]) * a

    solution = pynoza.Solution(
        max_order=order,
        wave_speed=c0
    )
    solution.recurse()
    solution.set_moments(
        current_moment=current_moment,
        charge_moment=None
    )
    e_field = solution.compute_e_field(
        x, y, z, t,
        hp, t,
        verbose=False
    )
    e_field_x = e_field[0, :, :, :, :]

    plt.plot(t, e_field_x.squeeze().T)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo_logo()
