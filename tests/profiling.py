#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022  Elias Le Boudec
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

import sys

if len(sys.argv) < 2:
    raise ValueError("Missing argument: filename")

filename = sys.argv[1]

def main():
    from test_EPFL_logo import test_solution
    import cProfile
    import pstats

    with cProfile.Profile() as pr:
        test_solution("logo")

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
    stats.dump_stats(filename=filename)


if __name__ == "__main__":
    main()
