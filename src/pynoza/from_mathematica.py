import numpy as np
import json
from pathlib import Path
current_dir = Path(__file__).parent
data_path = current_dir / 'data' / 'map.json'


def _from_mathematica_tuple(t):
    return eval(t.replace("{", "(").replace("}", ")"))


def _from_mathematica(expr):
    return eval(expr.replace("Sqrt", "np.sqrt").replace("Pi", "np.pi"))


def _read_mathematica_export():
    mapping = dict()
    with open(data_path, "r") as fd:
        data = json.load(fd)
        for spherical_key, map in data.items():
            spherical_key = _from_mathematica_tuple(spherical_key)
            mapping[spherical_key] = dict()
            for cartesian_key, expression in map.items():
                cartesian_key = _from_mathematica_tuple(cartesian_key)
                mapping[spherical_key][cartesian_key] = _from_mathematica(expression)
    return mapping


SPHERICAL_TO_CARTESIAN = _read_mathematica_export()


if __name__ == "__main__":
    res = SPHERICAL_TO_CARTESIAN[(2, -1)]
    print(res)
    res = SPHERICAL_TO_CARTESIAN[(2, 0)]
    print(res)
    res = SPHERICAL_TO_CARTESIAN[(2, 1)]
    print(res)
    res = SPHERICAL_TO_CARTESIAN[(2, -2)]
    print(res)
    res = SPHERICAL_TO_CARTESIAN[(2, 2)]
    print(res)

