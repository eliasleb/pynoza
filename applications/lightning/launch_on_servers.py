from itertools import product
from random import shuffle, seed
from subprocess import run
from contextlib import redirect_stdout
import io


def get_all_args(**kwargs):
    all_args = []
    for args in product(*tuple(kwargs.values())):
        all_args.append(args)
    return all_args


def get_ips():
    ips = []
    with open("../../../../project/admin/infomaniak/ips.txt") as fd:
        while ip := fd.readline():
            ips.append(ip.strip())
    return ips


def run_remote_command(ip, command, verbose=True):
    if verbose:
        print(f"@{ip}: Running command {command:.30}...")
    ssh_command = f"ssh -i ~/.ssh/id_rsa ubuntu@{ip} '{command}'"
    return run(ssh_command, shell=True, capture_output=not verbose)


def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def launch(s=0, rm_cache=False):
    all_args = get_all_args(
        cases=("TL", "MTLL", "MTLE", "QUAD"),
        orders=(2, 4),
        n_points=(50, 60, 70),
        seeds=range(10)
    )
    seed(s)
    shuffle(all_args)
    ips = get_ips()
    all_args_split = split_list(all_args, len(ips))
    assert all_args == [item for sublist in all_args_split for item in sublist]
    for ip, sublist_args in zip(ips, all_args_split):
        commands = """
cd pynoza
git restore .
git pull
source venv/bin/activate
pip install .
cd applications/lightning
"""
        if rm_cache:
            commands += """rm function_cache/* || echo "Cache already empty"\n"""
        for case, order, n_point, s in sublist_args:
            noise_level = 0
            commands += f"""nohup python lightning.py --max_order {order} --n_points {n_point} --seed {s} """ \
                f"""--noise_level {noise_level} --case {case} --scale 1e9 > /dev/null 2>&1 &"""
            # f"""> ../../../git_ignore/lightning_inverse/opt_results/v{version}_max_order_{order}""" \
            # f"""_n_points_{n_point}_seed_{s}_noise_level_{noise_level}_case_{case}.txt > /dev/null 2>&1 &\n"""
        # commands += "exit\n"
        run_remote_command(ip, commands)


def monitor():
    for ip in get_ips():
        ret = run_remote_command(ip, "pgrep python | wc -l", verbose=False)
        n = int(ret.stdout)
        print(f"{ip:16} {n}")


if __name__ == "__main__":
    # launch()
    monitor()

