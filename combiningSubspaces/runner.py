import subprocess
from itertools import product

input = r"D:\Cloud\SVM\dg2_lin_100k_1k_b0_01_08.npz"

output = r"D:\Cloud\SVM\test-file.xlsx"

# Iiterating through the parameters for specific methods

#  SVC
C = [0.1, 0.5, 1.0]
kernels = ["linear"]

for _C, _kernel in product(C, kernels):
    subprocess.run([
        "python", "./experiment.py",
        "--model", "SVC",
        "--kernel", _kernel, 
        "--C", str(_C), 
        "--train_size", str(5000),
        "--test_size", str(5000),      
        "--input", input,
        "--output", output
    ], cwd = "combiningSubspaces", check = True)


# LinearSVC
C = [0.1, 1.0]
penaltys = ['l1', 'l2']

for _C, _penalty in product(C, penaltys):
    subprocess.run([
        "python", "./experiment.py",
        "--model", "LinearSVC",
        "--penalty", _penalty, 
        "--C", str(_C), 
        "--train_size", str(5000),
        "--test_size", str(5000),      
        "--input", input,
        "--output", output
    ], cwd = "combiningSubspaces", check = True)


# CombLinSVM
C = [0.1, 1.0]
splits = [1, 5, 10]

for _C, _splits in product(C, splits):
    subprocess.run([
        "python", "./experiment.py",
        "--model", "CombLinSVM",
        "--splits", str(_splits), 
        "--C", str(_C), 
        "--train_size", str(5000),
        "--test_size", str(5000),      
        "--input", input,
        "--output", output
    ], cwd = "combiningSubspaces", check = True)