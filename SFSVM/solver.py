import numpy as np
import mosek
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

    xx = np.asarray(mt.getxx(mosek.soltype.itr))
    suc = np.asarray(mt.getsuc(mosek.soltype.itr))
    slc = np.asarray(mt.getslc(mosek.soltype.itr))

    return {
        "optres": 0,
        "lambda": xx[:n],
        "r": -2 * suc[:k],
        "b": -suc[-1] - slc[-1],
        "delta": -suc[2 * k:2 * k + n],
    }


def calc_errors(alg_output, test_kernels, train_kernels):

    train_types = train_kernels["types"]
    test_types = test_kernels["types"]
    kernels = test_kernels["kernels"]

    n_train = len(train_types)
    n_test = len(test_types)
    n_kernels = kernels.shape[2]

    K = np.zeros((n_train, n_test))

    for i in range(n_kernels):
        K += kernels[:, :, i] * alg_output["r"][i]

    weights = alg_output["lambda"] * train_types
    scores = np.sum(weights[:, None] * K, axis=0) + alg_output["b"] 

    obtained_types = np.sign(scores)

    nP = np.sum(test_types == 1)
    nN = np.sum(test_types == -1)

    nFN = np.sum(
        (test_types == 1) & (obtained_types != test_types)
    )

    nFP = np.sum(
        (test_types == -1) & (obtained_types != test_types)
    )

    # в матлабе была ошибка
    nTP = nP - nFN
    nTN = nN - nFP

    error = {
        "nObj": n_test,
        "nP": nP,
        "nN": nN,
        "nFP": nFP,
        "nFN": nFN,
        "nTP": nTP,
        "nTN": nTN,
        "ER": (nFP + nFN) / n_test,
        "FNR": nFN / nP,
        "FPR": nFP / nN,
        "Precision": nTP / (nTP + nFP),
        "Recall": nTP / (nTP + nFN),
    }

    error["Fmeasure05"] = (
        2 * error["Precision"] * error["Recall"]
        / (error["Precision"] + error["Recall"])
    )

    return error, scores