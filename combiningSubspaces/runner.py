import os
import subprocess
from itertools import product

print(f"---START---")

## --- CV_experiment ---
dataFile = r"datasets\darwin\darwin.npz"
output = rf"D:\Cloud\SVM\test-CV5_{os.path.basename(dataFile)}.xlsx"

C = [0.1, 1, 10]
splits = [5, 10]
subtype = ['l1', 'l2']

for _C, _splits, _subtype in product(C, splits, subtype):
    subprocess.run([
        "python", "-m", "combiningSubspaces.CV_experiment",
        "--data", dataFile,
        "--C", str(_C),
        "--model", f"Comb-LSVC-{_subtype}",
        "--splits", str(_splits),
        "--output", output,
        "--no-std"
    ], cwd = ".", check = True)


# ## --- split_experiment ---
# trainFile = r"datasets\gisette\gisette_scale_tr.npz"
# testFile = r"datasets\gisette\gisette_scale_t.npz"
# output = rf"D:\Cloud\SVM\test-split_{os.path.basename(trainFile)}_{os.path.basename(testFile)}.xlsx"

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