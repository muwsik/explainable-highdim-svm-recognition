import numpy as np
from scipy.io import loadmat


def load_electrodes(
    filepath,
    n_electrodes,
    signals_per_electrode,
):
    X = np.loadtxt(filepath)

    if X.ndim != 2: 
        raise ValueError(
            f"A two‑dimensional matrix was expected, but received shape={X.shape}"
        )

    expected_columns = n_electrodes * signals_per_electrode

    if X.shape[1] != expected_columns:
        raise ValueError(
            f"Incorrect number of columns: {X.shape[1]}. "
            f"Expected {expected_columns} "
            f"({n_electrodes} × {signals_per_electrode})."
        )

    electrodes = [
        X[:, i * signals_per_electrode:(i + 1) * signals_per_electrode]
        for i in range(n_electrodes)
    ]

    return electrodes


def prepare_data(
    signals_path,
    classes_path,
    ind_electrodes,
    n_electrodes = 66,
    signals_per_electrode = 100,
):
    electrodes = load_electrodes(
        signals_path,
        n_electrodes,
        signals_per_electrode,
    )

    y = loadmat(classes_path)["classes"].ravel().copy()

    if ind_electrodes is None:
        ind_electrodes = range(1, n_electrodes + 1)

    X = np.hstack([
        electrodes[el - 1]
        for el in ind_electrodes
    ])

    if len(y) != X.shape[0]:
        raise ValueError(
            f"The number of class labels ({len(y)}) "
            f"does not match the number of objects ({X.shape[0]})."
        )

    return X, y