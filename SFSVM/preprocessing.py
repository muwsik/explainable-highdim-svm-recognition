import numpy as np
 

def load_electrodes(
    filepath,
    n_electrodes,
    signals_per_electrode,
):
    X = np.loadtxt(filepath)

    if X.ndim != 2: 
        raise ValueError(
            f"Ожидалась двумерная матрица, получено shape={X.shape}"
        )

    expected_columns = n_electrodes * signals_per_electrode

    if X.shape[1] != expected_columns:
        raise ValueError(
            f"Неверное количество столбцов: {X.shape[1]}. "
            f"Ожидалось {expected_columns} "
            f"({n_electrodes} × {signals_per_electrode})."
        )

    electrodes = [
        X[:, i * signals_per_electrode:(i + 1) * signals_per_electrode]
        for i in range(n_electrodes)
    ]

    return electrodes


def build_kernels(electrodes, classes, ind_electrodes,
                  train_indices, test_indices):

    train_types = classes[train_indices]
    test_types = classes[test_indices]

    n_train = len(train_indices)
    n_test = len(test_indices)
    n_signals = electrodes[0].shape[1]
    n_kernels = len(ind_electrodes) * n_signals

    kadd = np.eye(n_train) + np.outer(train_types, train_types)

    train_kernels = np.zeros((n_train, n_train, n_kernels))
    test_kernels = np.zeros((n_train, n_test, n_kernels))

    k = 0

    for el in ind_electrodes:
        signals = electrodes[el - 1]

        for j in range(n_signals):
            vec = signals[:, j]
            kernel = np.outer(vec, vec)

            train_kernels[:, :, k] = (
                kadd * kernel[np.ix_(train_indices, train_indices)]
            )
            test_kernels[:, :, k] = (
                kernel[np.ix_(train_indices, test_indices)]
            )

            k += 1

    return train_kernels, test_kernels, train_types, test_types