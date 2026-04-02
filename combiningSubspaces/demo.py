#%%
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import time

from sklearn.svm import SVC
from sklearn.svm import LinearSVC 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from combinedModel import combLinModel
from dataGenerator.sample import Sample


tempSeed = 42


def subspaceInfo(model, data, N):
    _, X_test, _, Y_test = train_test_split(
        data.X,
        data.Y,
        train_size = N,
        test_size = N,
        random_state = tempSeed
    )

    for i, submodel in enumerate(model.subspaceModels):
        timePredict = -time.time()
        myLabels = submodel[0].predict(X_test[:, model.subspaceIndex[i]])
        timePredict += time.time()

        print(f"""--{i}--
              acc test: {accuracy_score(Y_test, myLabels)},
              time train: {submodel[1]},
              time predict: {timePredict},
        """)


def fit_predict(model, data, N):
    X_train, X_test, Y_train, Y_test = train_test_split(
        data.X,
        data.Y,
        train_size = N,
        test_size = N,
        random_state = tempSeed
    )

    timeTrain = -time.time()
    model.fit(X_train, Y_train)
    timeTrain += time.time()

    timePredict = -time.time()
    myLabels = model.predict(X_test)
    timePredict += time.time()

    return model, {
            "acc test": accuracy_score(Y_test, myLabels),
            "time train": timeTrain,
            "time predict": timePredict,
            "confusion matrix test": confusion_matrix(Y_test, myLabels),
            "acc train": accuracy_score(Y_train, model.predict(X_train))
        }


# %%
tempDataset = Sample()
tempDataset.loadBin(r"D:\Загрузки\dg2_lin_100k_1k_b0_01_08.npz")


#%% Training and recognition for 5k objects
print(f"Training and recognition for 5k objects by SVC(kernel = 'linear')")

tempModel = SVC(kernel = 'linear')
_, info = fit_predict(tempModel, tempDataset, 5000)
print(info)


# %% combining hyperplanes built on subspaces
print(f"Training and recognition for 5k objects by combLinModel")

tempModel = combLinModel(1000, 10, 0)
_, info = fit_predict(tempModel, tempDataset, 5000)
print(info)

#%%

subspaceInfo(tempModel, tempDataset, 5000)


#%% Training and recognition for 10k objects
print(f"Training and recognition for 10k objects by SVC(kernel = 'linear')")

tempModel = SVC(kernel = 'linear')
_, info = fit_predict(tempModel, tempDataset, 10000)
print(info)

