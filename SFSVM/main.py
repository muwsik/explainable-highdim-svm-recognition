from scipy.io import loadmat
import numpy as np
from sklearn.metrics import roc_auc_score


from preprocessing import load_electrodes, build_kernels
from solver import kernel_task_SKM_MSK, kernel_SKM_MSK_origin, calc_errors


if __name__ == "__main__":
    electrodes = load_electrodes(
        r"D:\Projects\explainable-highdim-svm-recognition\SFSVM\data\signals_oliver_smooth_w11_scale.txt",
        66,
        100
    )

    classes = loadmat(r"D:\Projects\explainable-highdim-svm-recognition\SFSVM\data\classes.mat")["classes"].ravel()

    train_indices = np.arange(196)
    test_indices = np.arange(196, 755)

    ind_electrodes = [16, 26, 27, 28, 30, 33, 34, 37, 39, 42, 46, 53, 60]
    mu = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    C = 10

    for temp_mu in mu:
        print("\nSVM C = ", C)
        print("electrodes: ", ind_electrodes)
        print("sel = ", temp_mu)
        print("mu = sel^2/2 = ", temp_mu**2 / 2)

        train_kernels, test_kernels, train_types, test_types = build_kernels(
            electrodes,
            classes,
            ind_electrodes,
            train_indices,
            test_indices
        )

        task = kernel_task_SKM_MSK(1300, 196, C)

        result = kernel_SKM_MSK_origin(
            train_kernels,
            train_types,
            task,
            temp_mu**2
        )

        #print("sum(lambda) =", result["lambda"].sum())
        #print("max(lambda) =", result["lambda"].max())
        #print("nonzero lambda =", np.count_nonzero(np.abs(result["lambda"]) > 1e-10))
        #print("min(r) =", result["r"].min())
        #print("max(r) =", result["r"].max())
        #print("b =", result["b"])
        #print("delta min =", result["delta"].min())
        #print("delta max =", result["delta"].max())
        print("r1 = ", np.sum(np.abs(result["r"] - 1.0) < 1e-5))
        print("r001 = ", np.sum(result["r"] >= 0.01))

        error, scores = calc_errors(
            result,
            {
                "kernels": test_kernels,
                "types": test_types,
            },   
            {
                "kernels": train_kernels,
                "types": train_types,
            }
        )
    
        auc = roc_auc_score(
            test_types,
            scores
        )

        print("AUC = ", auc)