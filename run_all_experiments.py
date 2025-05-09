import subprocess
import itertools

scan_types = ["limited", "sparse"]
scan_nums = [64, 256]

args1 = [
    f".\\results\\{st}-{sn}\\" for st, sn in itertools.product(scan_types, scan_nums)
]
args2 = [str(ph_idx) for ph_idx in [4, 13, 16]]

all_args = list(itertools.product(args1, args2))
for arg1, arg2 in all_args:
    command = ["python", "ct_experiment.py", str(arg1), str(arg2)]
    print(f"Running command: {' '.join(command)}")
    subprocess.run(command)
