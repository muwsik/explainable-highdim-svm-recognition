#%%
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from combinedModel import combModel
from dataGenerator.sample import Sample


# %%
tempDataset = Sample()
tempDataset.loadBin(r"D:\Загрузки\dg2_lin_100k_1k_b0_01_08.npz")

tempX = tempDataset.X[::200, :]
tempY = tempDataset.Y[::200]

myModel = combModel(1000, 5, 1)
myModel.fit(tempX, tempY, lambda: SVC(kernel = 'linear'))

myScores, myLabels = myModel.predict(tempX)

accuracy = accuracy_score(tempY, myLabels)
f1 = f1_score(tempY, myLabels)
cm = confusion_matrix(tempY, myLabels)

print(f"accuracy: {accuracy};\n {cm}")