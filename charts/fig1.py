import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# Файл
# ============================================================

FILE = r"D:\Projects\explainable-highdim-svm-recognition\charts\res002_averaged.xlsx"


# ============================================================
# Настройки
# ============================================================

sizes = [20000, 30000, 40000]

size_labels = {
    20000: "20k",
    30000: "30k",
    40000: "40k"
}

methods = ["HP-DSC", "Bagging", "SVM"]


# ОДИН цвет метода используется в обоих подрисунках
colors = {
    "HP-DSC": "#1f77b4",
    "Bagging": "#ff7f0e",
    "SVM": "#2ca02c"
}


# Разные символы дополнительно различают методы
symbols = {
    "HP-DSC": "circle",
    "Bagging": "square",
    "SVM": "diamond"
}


# ============================================================
# Чтение листа
# ============================================================

def read_sheet(sheet):
    return pd.read_excel(FILE, sheet_name=sheet)


# ============================================================
# Определение регуляризации
# ============================================================

def get_regularization(model):

    model = str(model).lower()

    if "penalty='l1'" in model or 'penalty="l1"' in model:
        return "L1"

    return "L2"


# ============================================================
# HP-DSC
# ============================================================

hp_sheets = {
    20000: "HP-DSC-20k",
    30000: "HP-DSC-30k",
    40000: "HP-DSC-40k"
}

hp = []

for size, sheet in hp_sheets.items():

    df = read_sheet(sheet)

    # Только последовательный режим
    df = df[df["n_proc"] == 1].copy()

    df["train_size"] = size
    df["method"] = "HP-DSC"

    df["regularization"] = df["model"].apply(
        get_regularization
    )

    hp.append(df)

hp = pd.concat(
    hp,
    ignore_index=True
)


# ============================================================
# Bagging
# ============================================================

bagging_sheets = {
    20000: "Bagging-20k",
    30000: "Bagging-30k",
    40000: "Bagging-40k"
}

bagging = []

for size, sheet in bagging_sheets.items():

    df = read_sheet(sheet)

    df["train_size"] = size
    df["method"] = "Bagging"

    df["regularization"] = df["model"].apply(
        get_regularization
    )

    bagging.append(df)

bagging = pd.concat(
    bagging,
    ignore_index=True
)


# ============================================================
# SVM
# ============================================================

svm = read_sheet("SVM")

svm["train_size"] = svm["nTr_obj"]
svm["method"] = "SVM"

svm["regularization"] = svm["model"].apply(
    get_regularization
)


# ============================================================
# Выбор лучшей конфигурации
# ============================================================

hp_best = (
    hp
    .sort_values(
        "acc_tst_mean",
        ascending=False
    )
    .groupby(
        ["train_size", "regularization"],
        as_index=False
    )
    .first()
)


bagging_best = (
    bagging
    .sort_values(
        "acc_mean",
        ascending=False
    )
    .groupby(
        ["train_size", "regularization"],
        as_index=False
    )
    .first()
)


svm_best = (
    svm
    .sort_values(
        "acc_mean",
        ascending=False
    )
    .groupby(
        ["train_size", "regularization"],
        as_index=False
    )
    .first()
)


# ============================================================
# Единый формат
# ============================================================

hp_plot = hp_best[
    [
        "train_size",
        "regularization",
        "acc_tst_mean",
        "acc_tst_sd"
    ]
].rename(
    columns={
        "acc_tst_mean": "mean_acc",
        "acc_tst_sd": "sd_acc"
    }
)

hp_plot["method"] = "HP-DSC"


bagging_plot = bagging_best[
    [
        "train_size",
        "regularization",
        "acc_mean",
        "acc_sd"
    ]
].rename(
    columns={
        "acc_mean": "mean_acc",
        "acc_sd": "sd_acc"
    }
)

bagging_plot["method"] = "Bagging"


svm_plot = svm_best[
    [
        "train_size",
        "regularization",
        "acc_mean",
        "acc_sd"
    ]
].rename(
    columns={
        "acc_mean": "mean_acc",
        "acc_sd": "sd_acc"
    }
)

svm_plot["method"] = "SVM"


plot_data = pd.concat(
    [
        hp_plot,
        bagging_plot,
        svm_plot
    ],
    ignore_index=True
)

plot_data["size_label"] = plot_data[
    "train_size"
].map(size_labels)


# ============================================================
# Рисунок
# ============================================================

fig = make_subplots(
    rows=1,
    cols=2,

    subplot_titles=[
        "L1 regularization",
        "L2 regularization"
    ],

    horizontal_spacing=0.10
)


for col, regularization in enumerate(
    ["L1", "L2"],
    start=1
):

    for method in methods:

        d = (
            plot_data[
                (plot_data["regularization"] == regularization) &
                (plot_data["method"] == method)
            ]
            .sort_values("train_size")
        )

        fig.add_trace(

            go.Scatter(

                x=d["size_label"],
                y=d["mean_acc"],

                mode="lines+markers",

                name=method,

                legendgroup=method,

                # Легенда только один раз
                showlegend=(col == 1),

                line=dict(
                    color=colors[method],
                    width=2
                ),

                marker=dict(
                    color=colors[method],
                    symbol=symbols[method],
                    size=7
                ),

                error_y=dict(
                    type="data",
                    array=d["sd_acc"].fillna(0),

                    visible=True,

                    color=colors[method],

                    thickness=1.5,
                    width=6
                )
            ),

            row=1,
            col=col
        )


# ============================================================
# Автоматический масштаб Y
# ============================================================

# Находим общий диапазон данных + error bars.
y_low = (
    plot_data["mean_acc"] -
    plot_data["sd_acc"].fillna(0)
).min()

y_high = (
    plot_data["mean_acc"] +
    plot_data["sd_acc"].fillna(0)
).max()

# Небольшой запас сверху/снизу.
padding = (y_high - y_low) * 0.08

y_low -= padding
y_high += padding


fig.update_yaxes(
    title="Test accuracy",
    range=[y_low, y_high],
    tickformat=".3f",
    row=1,
    col=1
)

fig.update_yaxes(
    title=None,
    range=[y_low, y_high],
    tickformat=".3f",
    row=1,
    col=2
)


# ============================================================
# Оси X
# ============================================================

fig.update_xaxes(
    title="Training set size",
    row=1,
    col=1
)

fig.update_xaxes(
    title="Training set size",
    row=1,
    col=2
)


# ============================================================
# Оформление
# ============================================================

fig.update_layout(

    title="Classification accuracy versus training set size",

    template="plotly_white",

    # Увеличиваем физическую высоту рисунка,
    # а НЕ искусственно растягиваем диапазон Y.
    width=1200,
    height=750,

    legend=dict(
        title="Method",

        orientation="h",

        yanchor="bottom",
        y=1.03,

        xanchor="left",
        x=0
    ),

    margin=dict(
        l=80,
        r=40,
        t=110,
        b=80
    )
)

fig.show()

fig.write_image(
    r"D:\Projects\explainable-highdim-svm-recognition\charts\figure_1.svg",
    width=1200,
    height=750
)