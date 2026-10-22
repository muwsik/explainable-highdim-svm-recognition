import numpy as np
from sklearn.metrics import roc_auc_score

from preprocessing import load_matlab_data, preprocess_raw_signals, split_data
from SFSVM import SFSVM


if __name__ == "__main__":
    ## Parameters
    # Indices in Matlab start from 1
    # They are sequential numbers: first is 1, last is 66
    ind_electrodes = [16, 26, 27, 28, 30, 33, 34, 37, 39, 42, 46, 53, 60]
    sel = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    C = 100
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

    for temp_sel in sel:
        print(f"\nSVM C = {C}")
        print(f"electrodes: {ind_electrodes}")
        print(f"sel = {temp_sel}")
        print(f"mu = sel^2/2 = {temp_sel**2 / 2 :.3f}")

        model = SFSVM(temp_sel**2, C)

        model.fit(x_train, y_train)

        print("r1 =", np.sum(np.abs(model.fit_result["r"] - 1.0) < 1e-5))
        print("r001 =", np.sum(model.fit_result["r"] >= 0.01))

        predict_scores = model.decision_function(x_test)
    
        auc = roc_auc_score(y_test, predict_scores)
        print(f"AUC = {auc:.4f}")