
# %%
import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import time

from sklearn.svm import SVC, LinearSVC
from combinedModel import combLinModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from dataGenerator.sample import Sample


# run one experiment
if __name__ == "__main__":

    # 1. Configuration command line arguments
    parser = argparse.ArgumentParser()

    # 1.1 General 
    parser.add_argument("--input", type = str, required = True,
        help = "Path to dataset (.npz)")
    parser.add_argument("--output", type = str, required = True,
        help = "Output Excel file")
    parser.add_argument("--model", type = str, required = True,
        choices=  ["SVC", "LinearSVC", "CombLinSVM"],
        help = "Model type")    
    parser.add_argument("--train_size", type = int, default = 5000,
        help = "Train part size")
    parser.add_argument("--test_size", type = int, default = 5000,
        help = "Test part size")
    parser.add_argument("--seed", type = int, default = 42,
        help = "Random seed (for splitting full dataset)")
    parser.add_argument("--C", type = float, default = 1.0,
        help = "Regularization SVM parameter")

    # 1.2 SVC
    parser.add_argument("--kernel", type = str, default = None,
        choices = ["linear", "rbf"],
        help = "Kernel type (for SVC)")

    # 1.3 LinearSVC
    parser.add_argument("--penalty", type = str, default = None,
        choices = ["l1", "l2"],
        help = "Penalty type (for LinearSVC)")

    # 1.4 CombLinSVM 
    parser.add_argument("--splits", type = int, default = None,
        help = "Number of subspaces (for CombLinSVM)")

    # Command line arguments
    args = parser.parse_args()


    # 2. Main experiment logic
    # 2.1 Full dataset load
    tempDataset = Sample()
    tempDataset.loadBin(args.input)
    print(f"Dataset '{args.input}' loded.\n\tParameters of dataset generation: {tempDataset.params}")
#%%
    # 2.2 Split full dataset on parts
    X_train, X_test, Y_train, Y_test = train_test_split(
        tempDataset.X,
        tempDataset.Y,
        train_size = args.train_size,  
        test_size = args.test_size,
        random_state = args.seed
    )

    # 2.3 Model for experiment
    if args.model == "SVC":    
        model = SVC(C = args.C, kernel = args.kernel)
    elif args.model == "LinearSVC":    
        model = LinearSVC(C = args.C, penalty = args.penalty, dual = False)
    elif args.model == "CombLinSVM":
        model = combLinModel(numSplits = args.splits,
            baseModel = lambda: SVC(C = args.C, kernel = 'linear')) # TODO: type model switch
    else:
        raise ValueError("Unknown model")

    # 2.4 Training
    print(f"\tTraining model...")
    timeTrain = -time.time()
    model.fit(X_train, Y_train)
    timeTrain += time.time()

    # 2.5 Predicting
    print(f"\tPredicting...")
    timePredict = -time.time()
    myLabels = model.predict(X_test)
    timePredict += time.time()

    results = {
        "acc(test)": accuracy_score(Y_test, myLabels),
        "acc(train)": accuracy_score(Y_train, model.predict(X_train)),
        "time(train)": timeTrain,
        "time(predict)": timePredict,
    }

    # for parameters that can be equal to None
    check = lambda x: x if x is not None else "---"

    params = {
        # general
        "train_size": args.train_size,
        "test_size": args.test_size,
        "model": args.model,
        "C": args.C,

        # SVC
        "kernel": check(args.kernel),
        
        # LinearSVC
        "penalty": check(args.penalty),

        # CombLinSVM
        "splits": check(args.splits)

    }

    # 3. Writing results to Excel file
    row = {**params, **results}
    row['dataset'] = os.path.basename(args.input)
    row["id"] = time.time()
    row["seed"] = args.seed
    
    df = pd.DataFrame([row])
    if os.path.exists(args.output):
        fileDF = pd.read_excel(args.output, sheet_name = "runs")
        df = pd.concat([fileDF, df], ignore_index = True)

    metricCols = ["acc(test)", "acc(train)", "time(train)", "time(predict)"]

    paramCols = [
        c for c in df.columns
            if c not in metricCols + ['id', 'seed', 'dataset']
    ]

    grouped = df.groupby(paramCols)

    aggResults = {}
    for col in metricCols:
        aggResults[col + "_mean"] = (col, "mean")
        aggResults[col + "_std"] = (col, "std")

    aggDF = grouped.agg(**aggResults).reset_index()

    with pd.ExcelWriter(args.output, engine = "openpyxl") as writer:
        df.to_excel(writer, sheet_name = "runs", index = False)
        aggDF.to_excel(writer, sheet_name = "aggregated", index = False)

    print(f"\tResults write in file '{os.path.basename(args.output)}'")