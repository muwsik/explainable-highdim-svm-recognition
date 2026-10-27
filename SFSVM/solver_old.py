# License file needs to be placed in "C:\Users\<user>\mosek\mosek.lic"
# Trial license on cite https://www.mosek.com/products/trial/
import mosek

import numpy as np
from scipy.sparse import lil_matrix


def kernel_task_SKM_MSK(k, n, C):
    nnz_q = n * (n + 1) // 2

    qcsubi = np.tile(
        np.concatenate([np.arange(j, n) for j in range(n)]),
        k
    )

    qcsubj = np.tile(
        np.repeat(np.arange(n), np.arange(n, 0, -1)),
        k
    )

    qcsubk = np.repeat(np.arange(k), nnz_q)

    ind = qcsubi + n * qcsubj + n * n * qcsubk

    c = np.ones(n + k)
    c[:n] *= C
    c[n:] /= 2

    a = lil_matrix((2 * k + n, n + k))
    a[:k, n:] = np.eye(k)
    a[k:2*k, n:] = np.eye(k)
    a[2*k:, :n] = np.eye(n)

    return {
        "c": c,
        "ind": ind,
        "qcsubk": qcsubk,
        "qcsubi": qcsubi,
        "qcsubj": qcsubj,
        "a": a.tocsr(),
        "buc_mu": np.ones(k) / 2,

        "buc": np.r_[
            np.zeros(k),
            np.full(n, C / 2),
            0.0
        ],

        "blc": np.r_[
            np.full(k, -np.inf),
            np.full(k, -np.inf),
            np.zeros(n)
        ],
    }


def kernel_SKM_MSK_origin(data, labels, task, mu):
    n, _, k = data.shape

    qcval = data.ravel(order="F")[task["ind"]]

    buc = np.r_[
        mu * task["buc_mu"],
        task["buc"]
    ]

    blc = np.r_[
        task["blc"],
        0.0
    ]

    A = task["a"].tolil()
    A.resize(A.shape[0] + 1, A.shape[1])
    A[-1, :n] = labels
    A = A.tocsr()

    numvar = n + k
    numcon = A.shape[0]

    env = mosek.Env()
    mt = env.Task(0, 0)

    mt.appendvars(numvar)
    mt.appendcons(numcon)

    # Свободные переменные
    mt.putvarboundlist(
        range(numvar),
        [mosek.boundkey.fr] * numvar,
        [-np.inf] * numvar,
        [np.inf] * numvar
    )

    # Целевая функция
    mt.putclist(
        range(numvar),
        task["c"].tolist()
    )
    mt.putobjsense(mosek.objsense.maximize)

    # Линейные ограничения
    for i in range(numcon):

        start = A.indptr[i]
        end = A.indptr[i + 1]

        if start < end:
            mt.putarow(
                i,
                A.indices[start:end].tolist(),
                A.data[start:end].tolist()
            )

        if np.isneginf(blc[i]) and np.isposinf(buc[i]):
            bkc = mosek.boundkey.fr
        elif np.isneginf(blc[i]):
            bkc = mosek.boundkey.up
        elif np.isposinf(buc[i]):
            bkc = mosek.boundkey.lo
        elif blc[i] == buc[i]:
            bkc = mosek.boundkey.fx
        else:
            bkc = mosek.boundkey.ra

        mt.putconbound(
            i,
            bkc,
            float(blc[i]),
            float(buc[i])
        )

    # Все квадратичные ограничения одним вызовом
    mt.putqcon(
        task["qcsubk"].astype(np.int32),
        task["qcsubi"].astype(np.int32),
        task["qcsubj"].astype(np.int32),
        qcval
    )

    mt.optimize()

    solsta = mt.getsolsta(mosek.soltype.itr)

    if solsta != mosek.solsta.optimal:
        raise RuntimeError(
            f"MOSEK: The solution is not optimal."
            f"Status: {solsta}"
        )

    xx = np.asarray(mt.getxx(mosek.soltype.itr))
    suc = np.asarray(mt.getsuc(mosek.soltype.itr))
    slc = np.asarray(mt.getslc(mosek.soltype.itr))

    return {
        "lambda": xx[:n],
        "r": -2 * suc[:k],
        "b": -suc[-1] - slc[-1],
        "delta": -suc[2 * k:2 * k + n],
    }


def build_kernels(X_left, X_right):
    n_left, n_features = X_left.shape
    n_right = X_right.shape[0]

    kernels = np.zeros((n_left, n_right, n_features))

    for j in range(n_features):
        kernels[:, :, j] = np.outer(
            X_left[:, j],
            X_right[:, j],
        )

    return kernels


if __name__ == "__main__":    
    import preprocessing
    import sklearn.metrics 

    ### Parameters
    ind_electrodes = [16, 26, 27, 28, 30, 33, 34, 37, 39, 42, 46, 53, 60]
    sel = 0
    C = 10

    print(f"\nSVM C = {C}")
    print(f"electrodes: {ind_electrodes}")
    print(f"sel = {sel}")
    print(f"mu = sel^2/2 = {sel**2 / 2:.3f}")
    

    ### Preprocessing
    x_data, y_data = preprocessing.prepare_data(
        r"SFSVM\data\signals_oliver_smooth_w11_scale.txt",
        r"SFSVM\data\classes.mat",
        ind_electrodes
    )    
    x_train = x_data[:196]; y_train = y_data[:196]
    x_test = x_data[196:]; y_test = y_data[196:]


    ### fit
    kernels = build_kernels(x_train, x_train)
    kadd = np.eye(len(y_train)) + np.outer(y_train, y_train)
    train_kernels = kernels * kadd[:, :, None] 

    # n - objects, k - features
    n, k = x_train.shape
    empty_task = kernel_task_SKM_MSK(k, n, C)

    fit_result = kernel_SKM_MSK_origin(train_kernels, y_train,
                                        empty_task, sel**2)

    print("r1 =", np.sum(np.abs(fit_result["r"] - 1.0) < 1e-5))
    print("r001 =", np.sum(fit_result["r"] >= 0.01))


    ### decision_function
    kernels = build_kernels(x_train, x_test)

    K = np.zeros((x_train.shape[0], x_test.shape[0]))
    for i in range(kernels.shape[2]):
        K += kernels[:, :, i] * fit_result["r"][i]

    weights = fit_result["lambda"] * y_train
    kernel_part = np.sum(weights[:, None] * K, axis=0)
    predict_scores = kernel_part + fit_result["b"]
    y_predict = np.sign(predict_scores)
    
    auc = sklearn.metrics.roc_auc_score(y_test, predict_scores)
    print(f"AUC = {auc:.4f}")
    
    # cm = sklearn.metrics.confusion_matrix(y_test, y_predict)
    # print(f"{cm}")

    # rep = sklearn.metrics.classification_report(y_test, y_predict)
    # print(f"{rep}")