import argparse
import os

import pandas as pd
import time

from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from combiningSubspaces.combinedModel import combLinModel
from dataGenerator.sample import Sample

# run one experiment
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
        choices=  ["SVC", "LinearSVC", "CombLinSVM"],
        help = "Model type")    
    
    # 1.2 Base SVM parameters
    parser.add_argument("--C", type = float, default = 1.0,
        help = "Regularization SVM parameter")

    # 1.3 SVC model parameters
    parser.add_argument("--kernel", type = str, default = None,
        choices = ["linear", "rbf"],
        help = "Kernel type (for SVC)")

    # 1.4 LinearSVC model parameters
    parser.add_argument("--penalty", type = str, default = None,
        choices = ["l1", "l2"],
        help = "Penalty type (for LinearSVC)")

    # 1.5 CombLinSVM model parameters
    parser.add_argument("--splits", type = int, default = None,
        help = "Number of subspaces (for CombLinSVM)")

    # 1.Final 
    args = parser.parse_args()

    # For parameters that can be equal to None
    check = lambda x: x if x is not None else "---"
    
    params = {
        # general        
        "id": time.time(),
        "train": os.path.basename(args.train),
        "test": os.path.basename(args.test),
        "model": args.model,
        "C": args.C,

        # SVC
        "kernel": check(args.kernel),
        
        # LinearSVC
        "penalty": check(args.penalty),

        # CombLinSVM
        "splits": check(args.splits)
    }
    print(params)


    # 2. Main experiment logic
    # 2.1 Train dataset load
    trainDataset = Sample.fromBin(args.train)
    print(f"Loded train dataset '{args.train}'.")
    #print(f"Parameters of dataset generation: {trainDataset.params}")
    
    # standardization 
    scaler = StandardScaler()
    trainDataset.X = scaler.fit_transform(trainDataset.X)
    print(f"Train dataset is standardized.")

    # 2.2 Test dataset load
    testDataset = Sample.fromBin(args.test)
    print(f"Test dataset '{args.test}' loded.")
    #print(f"Parameters of dataset generation: {testDataset.params}")

    # standardization 
    testDataset.X = scaler.transform(testDataset.X)
    print(f"Test dataset is standardized.")

    # 2.3 Model for experiment
    if args.model == "SVC":    
        model = SVC(C = args.C, kernel = args.kernel, verbose = True)
    elif args.model == "LinearSVC":    
        model = LinearSVC(C = args.C, penalty = args.penalty, dual = False)
    elif args.model == "CombLinSVM":
        model = combLinModel(numSplits = args.splits,
            baseModel = lambda: LinearSVC(C = args.C, penalty = args.penalty, dual = False, verbose = True))
        raise ValueError("Unknown model")

    # 2.4 Training
    print(f"\tTraining model...")
    timeTrain = -time.time()
    model.fit(trainDataset.X, trainDataset.Y)
    timeTrain += time.time()

    # 2.5 Predicting
    print(f"\tPredicting...")
    timePredict = -time.time()
    myLabels = model.predict(testDataset.X)
    timePredict += time.time()

    # 2.Final 
    results = {
        "acc(test)": accuracy_score(testDataset.Y, myLabels),
        "acc(train)": accuracy_score(trainDataset.Y, model.predict(trainDataset.X)),
        "time(train)": timeTrain,
        "time(predict)": timePredict,
    }


    # 3. Writing results to Excel file
    df = pd.DataFrame([{**params, **results}])
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