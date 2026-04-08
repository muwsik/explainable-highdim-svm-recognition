import os
import subprocess
from itertools import product

trainFile = r"D:\datasets\train-10k-1k.npz"
testFile = r"D:\datasets\test-10k-1k.npz"

output = rf"D:\Cloud\SVM\{os.path.basename(trainFile)}__{os.path.basename(testFile)}.xlsx"

# Iiterating through the parameters for specific methods

print(f"---START---")


# SVC
C = [0.1, 1, 10]
kernels = ["linear"]

for _C, _kernel in product(C, kernels):
    subprocess.run([
        "python", "-m", "combiningSubspaces.experiment",
        "--train", trainFile,
        "--test", testFile,
        "--C", str(_C),   
        "--model", "SVC",
        "--kernel", _kernel, 
        "--output", output
    ], cwd = ".", check = True)

# CombLinSVM
C = [0.1, 1, 10]
splits = [1, 10, 50]

for _C, _splits in product(C, splits):
    subprocess.run([
        "python", "-m", "combiningSubspaces.experiment",
        "--train", trainFile,
        "--test", testFile,
        "--C", str(_C),   
        "--model", "CombLinSVM-LSVC",
        "--splits", str(_splits), 
        "--output", output
    ], cwd = ".", check = True)


# LinearSVC
C = [0.1, 1, 10]
penaltys = ['l1', 'l2']

for _C, _penalty in product(C, penaltys):
    subprocess.run([
        "python", "-m", "combiningSubspaces.experiment",
        "--train", trainFile,
        "--test", testFile,
        "--C", str(_C),   
        "--model", "LinearSVC",
        "--penalty", _penalty,          
        "--output", output
    ], cwd = ".", check = True)


print(f"---END---")