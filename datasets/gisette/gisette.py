import numpy as np
from sklearn.datasets import load_svmlight_file

# === 1. загрузка LIBSVM ===
X_sparse, y = load_svmlight_file(r"D:\Projects\explainable-highdim-svm-recognition\datasets\gisette\gisette_scale_tr")

# === 2. преобразование в dense ===
X = X_sparse.toarray().astype(np.float32)

# === 3. проверка классов ===
unique = np.unique(y)
print("Classes before:", unique)

# === 4. приведение к {-1, +1} ===
if set(unique) == {1, -1}:
    pass

y = y.astype(np.int32)

# === 5. финальная проверка ===
print("Shape:", X.shape)
print("Classes after:", np.unique(y))

# === 6. сохранение ===
np.savez(r"D:\Projects\explainable-highdim-svm-recognition\datasets\gisette\gisette_scale_tr.npz", X = X, Y = y)

print("Saved to *.npz")