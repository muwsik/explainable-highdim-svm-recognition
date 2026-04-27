import os
import subprocess
from itertools import product

print(f"---START---")

dataFile = r"D:\Projects\explainable-highdim-svm-recognition\datasets\darwin.npz"
output = rf"D:\Cloud\SVM\CV5_{os.path.basename(dataFile)}.xlsx"

C = [0.1, 1, 10]
splits = [1, 2, 5, 10]
subtype = ['l1', 'l2']

for _C, _splits, _subtype in product(C, splits, subtype):
    subprocess.run([
        "python", "-m", "combiningSubspaces.CV_experiment",
        "--data", dataFile,
        "--C", str(_C),
        "--model", f"Comb-LSVC-{_subtype}",
        "--splits", str(_splits),
        "--output", output
    ], cwd = ".", check = True)

print(f"---END---")