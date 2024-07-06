# Copyright (C) 2024  Elias Le Boudec
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.


from pynoza.helpers import levi_civita, get_magnetic_moment
import itertools
import numpy as np


def test_levi_civita():
    for i, j, k in itertools.product(range(1, 3 + 1), repeat=3):
        l = levi_civita(i, j, k, start_at_0=False)
        match (i, j, k):
            case (1, 2, 3):
                assert l == 1
            case (2, 3, 1):
                assert l == 1
            case (3, 1, 2):
                assert l == 1
            case (3, 2, 1):
                assert l == -1
            case (1, 3, 2):
                assert l == -1
            case (2, 1, 3):
                assert l == -1
            case _:
                assert l == 0


def test_magnetic_moment():
    max_order = 0
    current_moment = np.zeros((3, ) + (max_order + 2, ) * 3)
    current_moment[2, 0, 0, 0] = 1.
    magnetic_moment = get_magnetic_moment(current_moment)
    magnetic_moment_ref = np.zeros(current_moment.shape)
    magnetic_moment_ref[0, 0, 1, 0] = -1.
    magnetic_moment_ref[1, 1, 0, 0] = 1.
    assert np.allclose(magnetic_moment, magnetic_moment_ref)


def main():
    test_magnetic_moment()


if __name__ == "__main__":
    main()
