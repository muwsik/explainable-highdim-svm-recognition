import numpy as np
from scipy.io import loadmat

def split_data(signals, labels, ind_electrodes, train_end = 196):
    # Electrode selection: the numbers are set in MATLAB style (from 1)
    selected = signals[:, :, np.asarray(ind_electrodes) - 1]

    # One feature vector per object
    X = selected.reshape(selected.shape[0], -1)
    y = np.asarray(labels).ravel()

    # Division by MATLAB indexes: 1:196 and 197:755
    X_train, X_test = X[:train_end], X[train_end:]
    y_train, y_test = y[:train_end], y[train_end:]

    return X_train, X_test, y_train, y_test


# preparing raw data from .mat file
def load_matlab_data(
    matlab_file_path,
    classes_path,
    n_objects = 755,
    n_electrodes = 66,
    signals_per_electrode = 1100,
):
    # loading raw data from .mat file
    data = loadmat(matlab_file_path)
    train_target = data["traint"]
    train_no_target = data["trainnt"]
    test_target = data["testt"]
    test_no_target = data["testnt"]

    raw_signals = np.vstack([
        train_target,
        train_no_target,
        test_target,
        test_no_target,
    ])
    raw_signals = raw_signals.reshape(n_objects, signals_per_electrode, n_electrodes)
    # result for Olever -> 755 objects with 1100 features for everyone 66 electrodes 
    
    labels = loadmat(classes_path)["classes"].ravel().copy()

    if len(labels) != n_objects:
        raise ValueError(
            f"The number of class labels ({len(labels)}) "
            f"does not match the number of objects ({n_objects})."
        )

    return raw_signals, labels



def preprocess_raw_signals(
    raw_signal,
    thin = True,
    step = 11,
    smooth = True,
    window = 11,
    scale = False,
):
    # Формат: (объекты, временные отсчёты, электроды)
    result = raw_signal.astype(float, copy = True)

    if result.ndim != 3:
        raise ValueError(
            f"Expected a 3D array (objects, time, electrodes), "
            f"got shape = {result.shape}"
        )

    # Прореживание временной оси
    if thin:
        if result.shape[1] % step != 0:
            raise ValueError(
                f"Signal length {result.shape[1]} "
                f"is not divisible by step = {step}"
            )
        result = result[:, ::step, :]

    # Сглаживание каждого электрода по временной оси
    if smooth:
        n = result.shape[1]

        if window % 2 == 0:
            raise ValueError("'window' must be odd")
        if window > n:
            raise ValueError(
                f"Window {window} exceeds signal length {n}"
            )

        hw = (window - 1) // 2
        smoothed = np.empty_like(result)

        # Граничные значения
        smoothed[:, :hw, :] = np.mean(
            result[:, :window, :], axis=1
        )[:, None, :]
        smoothed[:, n-hw:, :] = np.mean(
            result[:, n-window:, :], axis=1
        )[:, None, :]

        # Скользящее среднее
        for i in range(hw, n - hw):
            smoothed[:, i, :] = np.mean(
                result[:, i-hw:i+hw+1, :], axis=1
            )

        result = smoothed

    # Масштабирование отдельно для каждого объекта и электрода
    if scale:
        minv = np.min(result, axis=1, keepdims=True)
        maxv = np.max(result, axis=1, keepdims=True)
        denom = maxv - minv

        result = np.divide(
            result - minv,
            denom,
            out = np.zeros_like(result),
            where = denom != 0,
        )

    return result



# loading preprocessed data from .txt file
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

# preparing preprocessed data from .txt file
def prepare_txt_data(
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