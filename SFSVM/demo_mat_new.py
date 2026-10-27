import numpy as np
from sklearn.metrics import roc_auc_score, classification_report

from preprocessing import load_matlab_data, preprocess_raw_signals, split_data
from SFSVM import SFSVM_new

if __name__ == "__main__":
    ## Parameters
    # Indices in Matlab start from 1
    # They are sequential numbers: first is 1, last is 66
    ind_electrodes = [16, 26, 27, 28, 30, 33, 34, 37, 39, 42, 46, 53, 60]
    MU = [0, 0.0025, 0.005, 0.0075, 0.01, 0.025] 
    C = 10
    ##

    # Loading raw data
    raw_signals, labels = load_matlab_data(
        matlab_file_path = "SFSVM/data/workspace_oliver.mat",
        classes_path = "SFSVM/data/classes.mat",
    )

    # Preprocessing
    signals = preprocess_raw_signals(
        raw_signals,
        thin = True,
        step = 11,
        smooth = True,
        window = 11,
        scale = False,
    )

    print("Raw signals:", raw_signals.shape)
    print("Preprocessed signals:", signals.shape)
    print("Labels:", labels.shape)    

    x_train, x_test, y_train, y_test = split_data(signals, labels, ind_electrodes)

    for _mu in MU:        
        print("\n=== NEW Experiment ===")
        print(f"\nSVM C = {C}")
        print(f"electrodes: {ind_electrodes}")
        print(f"mu = {_mu}")

        model = SFSVM_new(_mu, C)

        model.fit(x_train, y_train)

        predict_scores = model.decision_function(x_test)
        predict_labels = model.predict(x_test)

        auc = roc_auc_score(y_test, predict_scores)
        report = classification_report(y_test, predict_labels, labels = [1, -1])

        print("\nClassification:")
        print("AUC:", f"{auc:.4f}")
        print('\n'.join(report.splitlines()[:6]))  