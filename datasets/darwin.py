import pandas as pd
import numpy as np

df = pd.read_csv(r"D:\Projects\explainable-highdim-svm-recognition\datasets\data.csv")

# удалить ID
df = df.iloc[:, 1:]

# target — последняя колонка
target_col = df.columns[-1]

X = df.drop(columns=[target_col]).values
y = df[target_col].values

# кодирование классов
y = np.where(y == "P", 1, -1)

# типы
X = X.astype(np.float32)
y = y.astype(np.int32)

# проверка
print(X.shape)

unique, counts = np.unique(y, return_counts=True)
print(dict(zip(unique, counts)))

# сохранить
np.savez(r"D:\Projects\explainable-highdim-svm-recognition\datasets\darwin.npz", X=X, Y=y)