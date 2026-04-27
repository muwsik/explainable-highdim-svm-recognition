import argparse
import os

import numpy as np
import pandas as pd
import time

from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from combiningSubspaces.combinedModel import combLinModel
from dataGenerator.sample import Sample


# for data standardization
_flagStandardization = True

# for output of service information
_verbose = True

# run one experiment
if __name__ == "__main__":


    # 1. Configuration command line arguments
    parser = argparse.ArgumentParser()

    # 1.1 General app parameters
    parser.add_argument("--data", type = str, required = True,
        help = "Path to full dataset (.npz)")
    parser.add_argument("--output", type = str, required = True,
        help = "Output Excel file")
    parser.add_argument("--model", type = str, required = True,
        choices=  ["SVC-linear", "Comb-LSVC-l1", "Comb-LSVC-l2"],
        help = "Model type")    
    
    # 1.2 Base SVM parameters
    parser.add_argument("--C", type = float, default = 1.0,
        help = "Regularization SVM parameter")

    # 1.3 CombLinSVM model parameters
    parser.add_argument("--splits", type = int, default = None,
        help = "Number of subspaces (for Comb-LSVC-l1, Comb-LSVC-l2)")

    # 1.Final 
    args = parser.parse_args()

    # For parameters that can be equal to None
    check = lambda x: x if x is not None else "---"
    
    params = {
        # general        
        "id": time.time(),
        "seed": np.random.randint(0, 2**31 - 1),
        "data": os.path.basename(args.data),
        "model": args.model,
        "C": args.C,

        # CombLinSVM
        "splits": check(args.splits)
    }
    print(f"\nExperiment params: {params}")


    # 2. Main experiment logic
    # 2.1 Dataset load
    dataset = Sample.fromBin(args.data)
    print(f"Loded dataset '{args.data}'.")
    
    # 2.2 Split dataset
    fullResults = []

    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = params['seed'])
    for trainIndex, testIndex in skf.split(dataset.X, dataset.Y):
        trainSet = Sample(dataset.X[trainIndex], dataset.Y[trainIndex]) 
        testSet = Sample(dataset.X[testIndex], dataset.Y[testIndex]) 

        if (_flagStandardization):
            scaler = StandardScaler()
            trainSet.X = scaler.fit_transform(trainSet.X)
            testSet.X = scaler.transform(testSet.X)
            print(f"Fold standardized")

        # 2.3 Model for experiment
        if args.model == "SVC-linear":    
            model = SVC(
                C = args.C,
                kernel = 'linear',
                verbose = _verbose
            )
        elif args.model == "Comb-LSVC-l1":
            model = combLinModel(
                numSplits = args.splits,
                baseModel = lambda: LinearSVC(C = args.C, penalty = 'l1', dual = False, verbose = _verbose)
            )
        elif args.model == "Comb-LSVC-l2":
            model = combLinModel(
                numSplits = args.splits,
                baseModel = lambda: LinearSVC(C = args.C, penalty = 'l2', dual = 'auto', verbose = _verbose)
            )
        else:
            raise ValueError("Unknown model!")

        # 2.4 Training
        print(f"Training model {args.model}")
        timeTrain = -time.time()
        model.fit(trainSet.X, trainSet.Y)
        timeTrain += time.time()

        # 2.5 Predicting
        print(f"Predicting...")
        timePredict = -time.time()
        myLabels = model.predict(testSet.X)
        timePredict += time.time()

        # 2.6 Quality matrix
        TN, FP, FN, TP = confusion_matrix(testSet.Y, myLabels, labels = [-1, 1]).ravel()

        # 2.Final 
        tempResults = {
            "acc(test)": accuracy_score(testSet.Y, myLabels),
            "auc(test)": roc_auc_score(testSet.Y, model.decision_function(testSet.X)),

            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,

            "acc(train)": accuracy_score(trainSet.Y, model.predict(trainSet.X)),
            "time(train)": timeTrain,
            "time(predict)": timePredict,
        }

        fullResults.append({**params, **tempResults})


    # 3. Writing results to Excel file
    df = pd.DataFrame(fullResults)
    if os.path.exists(args.output):
        fileDF = pd.read_excel(args.output, sheet_name = "runs")
        df = pd.concat([fileDF, df], ignore_index = True)

    metricCols = list(tempResults.keys())

    paramCols = [
        c for c in df.columns
            if c not in metricCols + ['id', 'seed', 'data']
    ]

    grouped = df.groupby(paramCols)

    aggResults = {}
    for col in metricCols:
        aggResults[col + "_mean"] = (col, "mean")
        aggResults[col + "_std"] = (col, "std")
    
    aggResults["runs"] = (metricCols[0], "count")

    aggDF = grouped.agg(**aggResults).reset_index()

    with pd.ExcelWriter(args.output, engine = "openpyxl") as writer:
        df.to_excel(writer, sheet_name = "runs", index = False)
        aggDF.to_excel(writer, sheet_name = "aggregated", index = False)

    print(f"Results write in file '{os.path.basename(args.output)}'\n")