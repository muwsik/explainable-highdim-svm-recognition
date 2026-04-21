import os
import subprocess
from itertools import product

trainFile = r"D:\datasets\mc-train-5k-f5000-i5000-r0-l0.npz"
testFile = r"D:\datasets\mc-test-5k-f5000-i5000-r0-l0.npz"

output = rf"D:\Cloud\SVM\b_new_{os.path.basename(trainFile)}__{os.path.basename(testFile)}.xlsx"

# Iiterating through the parameters for specific methods

print(f"---START---")

# # SVC
# C = [0.001, 0.01, 0.1, 1]
# kernels = ["linear"]

# for _C, _kernel in product(C, kernels):
#     subprocess.run([
#         "python", "-m", "combiningSubspaces.experiment",
#         "--train", trainFile,
#         "--test", testFile,
#         "--C", str(_C),   
#         "--model", "SVC",
#         "--kernel", _kernel, 
#         "--output", output
#     ], cwd = ".", check = True)


# # LinearSVC
# C = [0.01, 0.1, 1, 10]  
# penaltys = ['l1']

# for _C, _penalty in product(C, penaltys):
#     subprocess.run([
#         "python", "-m", "combiningSubspaces.experiment",
#         "--train", trainFile,
#         "--test", testFile,
#         "--C", str(_C),   
#         "--model", "LinearSVC",
#         "--penalty", _penalty,          
#         "--output", output
#     ], cwd = ".", check = True)


# CombLinSVM
C = [0.01, 0.1, 1] 
splits = [1, 10, 100, 1000]
subtype = ['l1', 'l2']

for _C, _splits, _subtype in product(C, splits, subtype):
    subprocess.run([
        "python", "-m", "combiningSubspaces.experiment",
        "--train", trainFile,
        "--test", testFile,
        "--C", str(_C),   
        "--model", f"CombLinSVM-LSVC-{_subtype}",
        "--splits", str(_splits), 
        "--output", output
    ], cwd = ".", check = True)


print(f"---END---")