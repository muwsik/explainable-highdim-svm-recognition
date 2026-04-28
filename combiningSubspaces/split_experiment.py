import argparse
import os

import numpy as np
import pandas as pd
import time

from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

from combiningSubspaces.combinedModel import combLinModel
from dataGenerator.sample import Sample


# for data standardization
_flagStandardization = True

# for output of service information
_verbose = True


if __name__ == "__main__":
    # 1. Configuration command line arguments
    parser = argparse.ArgumentParser()

    # 1.1 General app parameters
    parser.add_argument("--train", type = str, required = True,
        help = "Path to train dataset (.npz)")
    parser.add_argument("--test", type = str, required = True,
        help = "Path to test dataset (.npz)")
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
        "train": os.path.basename(args.train),
        "test": os.path.basename(args.test),
        "model": args.model,
        "C": args.C,

        # CombLinSVM
        "splits": check(args.splits)
    }
    print(f"\nExperiment params: {params}")


    # 2. Main experiment logic
    # 2.1 Train dataset load
    trainDataset = Sample.fromBin(args.train)
    print(f"Loded train dataset '{params['train']}'.")
    
    # 2.2 Test dataset load
    testDataset = Sample.fromBin(args.test)
    print(f"Loded test dataset '{params['test']}'.")

    # standardization 
    if (_flagStandardization):
        scaler = StandardScaler()
        trainDataset.X = scaler.fit_transform(trainDataset.X)
        testDataset.X = scaler.transform(testDataset.X)
        print(f"Train and test datasets standardized")

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
            baseModel = lambda: LinearSVC(C = args.C, penalty = 'l1', dual = False, verbose = _verbose),
            seed = params['seed']
        )
    elif args.model == "Comb-LSVC-l2":
        model = combLinModel(
            numSplits = args.splits,
            baseModel = lambda: LinearSVC(C = args.C, penalty = 'l2', dual = 'auto', verbose = _verbose),            
            seed = params['seed']
        )
    else:
        raise ValueError("Unknown model!")

    # 2.4 Training
    print(f"Training model {args.model}")
    timeTrain = -time.time()
    model.fit(trainDataset.X, trainDataset.Y)
    timeTrain += time.time()

    # 2.5 Predicting
    print(f"Predicting...")
    timePredict = -time.time()
    myLabels = model.predict(testDataset.X)
    timePredict += time.time()

    # 2.6 Quality matrix
    TN, FP, FN, TP = confusion_matrix(testDataset.Y, myLabels, labels = [-1, 1]).ravel()

    # 2.Final 
    results = {
        "acc(test)": accuracy_score(testDataset.Y, myLabels),
        "auc(test)": roc_auc_score(testDataset.Y, model.decision_function(testDataset.X)),
        "acc(train)": accuracy_score(trainDataset.Y, model.predict(trainDataset.X)),

        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,

        "nonzero_f": np.sum(np.abs(model.coef_) > 1e-6),
        "time(train)": timeTrain,
        "time(predict)": timePredict,
    }


    # 3. Writing results to Excel file
    df = pd.DataFrame([{**params, **results}])
    if os.path.exists(args.output):
        fileDF = pd.read_excel(args.output, sheet_name = "runs")
        df = pd.concat([fileDF, df], ignore_index = True)

    metricCols = list(results.keys())

    paramCols = [
        c for c in df.columns
            if c not in metricCols + ['id', 'seed', 'data']
    ]

    grouped = df.groupby(paramCols)

    aggResults = {}
    aggResults["runs"] = (metricCols[0], "count")

    for col in metricCols:        
        aggResults[col + "_mean"] = (col, "mean")
        
    for col in metricCols:
        aggResults[col + "_std"] = (col, "std")    

    aggDF = grouped.agg(**aggResults).reset_index()

    with pd.ExcelWriter(args.output, engine = "openpyxl") as writer:
        df.to_excel(writer, sheet_name = "runs", index = False)
        aggDF.to_excel(writer, sheet_name = "aggregated", index = False)

    print(f"Results write in file '{os.path.basename(args.output)}'\n")