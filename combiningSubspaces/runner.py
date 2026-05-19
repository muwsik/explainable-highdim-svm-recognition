import os
import subprocess
from itertools import product
import numpy as np

print(f"---START---")

## --- synthetic_experiment ---
dataFile = r"D:\datasets\synthetic1-10k-f1000-i1000-r0-l0.npz"
output = rf"D:\Cloud\SVM\s1_{os.path.basename(dataFile)}.xlsx"

C = [0.1, 1, 10]
splits = [1, 2, 5, 10, 25, 50, 100]
subtype = ['l1', 'l2']

skf_seed = np.random.randint(0, 2**31 - 1)
for _C, _splits, _subtype in product(C, splits, subtype):
    subprocess.run([
        "python", "-m", "combiningSubspaces.synthetic_experiment",
        "--data", dataFile,
        "--C", str(_C),
        "--model", f"Comb-LSVC-{_subtype}",
        "--train-size", str(250),
        "--splits", str(_splits),
        "--skf-seed", str(skf_seed),
        "--output", output,
        "--no-std"
    ], cwd = ".", check = True)


# ## --- CV_experiment ---
# dataFile = r"D:\datasets\sync-kit-10k-f1000-i1000-r0-l0.npz"
# output = rf"D:\Cloud\SVM\test_{os.path.basename(dataFile)}.xlsx"

# C = [0.1, 1, 10]
# splits = [1]
# subtype = ['l1', 'l2']

# for _C, _splits, _subtype in product(C, splits, subtype):
#     subprocess.run([
#         "python", "-m", "combiningSubspaces.CV_experiment",
#         "--data", dataFile,
#         "--C", str(_C),
#         "--model", f"Comb-LSVC-{_subtype}",
#         "--splits", str(_splits),
#         "--folds", str(2),
#         "--output", output,
#         "--no-std"
#     ], cwd = ".", check = True)


# ## --- split_experiment ---
# trainFile = r"datasets\gisette\gisette_scale_tr.npz"
# testFile = r"datasets\gisette\gisette_scale_t.npz"
# output = rf"D:\Cloud\SVM\split_{os.path.basename(trainFile)}_{os.path.basename(testFile)}.xlsx"

# C = [0.1, 1, 10]
# splits = [10]
# subtype = ['l1', 'l2']

# for _C, _splits, _subtype in product(C, splits, subtype):
#     subprocess.run([
#         "python", "-m", "combiningSubspaces.split_experiment",
#         "--train", trainFile,
#         "--test", testFile,
#         "--C", str(_C),
#         "--model", f"Comb-LSVC-{_subtype}",
#         "--splits", str(_splits),
#         "--output", output,
#         "--no-std"
#     ], cwd = ".", check = True)


print(f"---END---")