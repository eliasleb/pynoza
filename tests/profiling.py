#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
