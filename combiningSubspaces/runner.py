import os
import subprocess
from itertools import product

trainFile = r"D:\datasets\ds-test-5k-5k-08-rnd-+15.npz"
testFile = r"D:\datasets\ds-train-10k-5k-08-rnd-+15.npz"

output = rf"D:\Cloud\SVM\{os.path.basename(trainFile)}__{os.path.basename(testFile)}.xlsx"

# Iiterating through the parameters for specific methods

print(f"---START---")

# #  SVC
# C = [1]
# kernels = ["linear"]

# for _C, _kernel in product(C, kernels):
#     subprocess.run([
#         "python", "./experiment.py",
#         "--train", trainFile,
#         "--test", testFile,
#         "--C", str(_C),   
#         "--model", "SVC",
#         "--kernel", _kernel, 
#         "--output", output
#     ], cwd = "combiningSubspaces", check = True)


# # LinearSVC
# C = [0.1]
# penaltys = ['l1']

# for _C, _penalty in product(C, penaltys):
#     subprocess.run([
#         "python", "./experiment.py",
#         "--train", trainFile,
#         "--test", testFile,
#         "--C", str(_C),   
#         "--model", "LinearSVC",
#         "--penalty", _penalty,          
#         "--output", output
#     ], cwd = "combiningSubspaces", check = True)


# CombLinSVM
C = [1]
splits = [2]

for _C, _splits in product(C, splits):
    subprocess.run([
        "python", "./experiment.py",
        "--train", trainFile,
        "--test", testFile,
        "--C", str(_C),   
        "--model", "CombLinSVM",
        "--splits", str(_splits), 
        "--output", output
    ], cwd = "combiningSubspaces", check = True)

print(f"---END---")