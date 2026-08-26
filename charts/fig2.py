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

HP_COLOR = "#1f77b4"
SVM_COLOR = "#2ca02c"


# ============================================================
# Регуляризация
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

    df = pd.read_excel(
        FILE,
        sheet_name=sheet
    )

    # Последовательная реализация
    df = df[df["n_proc"] == 1].copy()

    df["train_size"] = size

    df["regularization"] = df["model"].apply(
        get_regularization
    )

    hp.append(df)


hp = pd.concat(
    hp,
    ignore_index=True
)


# ============================================================
# SVM
# ============================================================

svm = pd.read_excel(
    FILE,
    sheet_name="SVM"
)

svm["train_size"] = svm["nTr_obj"]

svm["regularization"] = svm["model"].apply(
    get_regularization
)


# ============================================================
# Рисунок 2
# ============================================================

fig = make_subplots(

    rows=2,
    cols=3,

    # L1 / L2 по строкам,
    # 20k / 30k / 40k по столбцам
    subplot_titles=[
        "20k", "30k", "40k",
        "20k", "30k", "40k"
    ],

    vertical_spacing=0.12,
    horizontal_spacing=0.07
)


# ------------------------------------------------------------
# Добавляем подписи L1 / L2 слева от строк
# ------------------------------------------------------------

fig.add_annotation(
    text="<b>L1</b>",
    x=-0.065,
    y=0.75,
    xref="paper",
    yref="paper",
    textangle=-90,
    showarrow=False,
    font=dict(size=16)
)

fig.add_annotation(
    text="<b>L2</b>",
    x=-0.065,
    y=0.25,
    xref="paper",
    yref="paper",
    textangle=-90,
    showarrow=False,
    font=dict(size=16)
)


# ============================================================
# Заполняем панели
# ============================================================

for row, regularization in enumerate(
    ["L1", "L2"],
    start=1
):

    for col, size in enumerate(
        sizes,
        start=1
    ):

        # ----------------------------------------------------
        # HP-DSC
        # ----------------------------------------------------

        hp_data = (
            hp[
                (hp["train_size"] == size) &
                (hp["regularization"] == regularization)
            ]
            .sort_values("n_splits")
        )

        fig.add_trace(

            go.Scatter(

                x=hp_data["n_splits"],
                y=hp_data["acc_tst_mean"],

                mode="lines",

                name="HP-DSC",

                legendgroup="HP-DSC",

                showlegend=(
                    row == 1 and col == 1
                ),

                line=dict(
                    color=HP_COLOR,
                    width=2
                ),

                error_y=dict(
                    type="data",
                    array=hp_data["acc_tst_sd"].fillna(0),

                    visible=True,

                    color=HP_COLOR,

                    thickness=1.5,
                    width=6
                )
            ),

            row=row,
            col=col
        )


        # ----------------------------------------------------
        # SVM
        # ----------------------------------------------------

        svm_data = svm[
            (svm["train_size"] == size) &
            (svm["regularization"] == regularization)
        ]

        if len(svm_data) > 0:

            svm_accuracy = svm_data[
                "acc_mean"
            ].iloc[0]

            # Берём диапазон n_splits HP-DSC,
            # чтобы линия точно проходила через весь график.
            x_min = hp_data["n_splits"].min()
            x_max = hp_data["n_splits"].max()

            fig.add_trace(

                go.Scatter(

                    x=[
                        x_min,
                        x_max
                    ],

                    y=[
                        svm_accuracy,
                        svm_accuracy
                    ],

                    mode="lines",

                    name="SVM",

                    legendgroup="SVM",

                    showlegend=(
                        row == 1 and col == 1
                    ),

                    line=dict(
                        color=SVM_COLOR,
                        width=1.8,
                        dash="dash"
                    )
                ),

                row=row,
                col=col
            )


        # ----------------------------------------------------
        # Оси
        # ----------------------------------------------------

        fig.update_xaxes(
            title="n_splits",
            row=row,
            col=col
        )

        fig.update_yaxes(
            title="Test accuracy" if col == 1 else None,
            tickformat=".3f",
            row=row,
            col=col
        )


# ============================================================
# Общий диапазон Y
# ============================================================

# Чтобы L1 и L2 и все размеры можно было честно сравнивать,
# используем один диапазон Y для всего рисунка.

hp_values = hp[
    "acc_tst_mean"
].dropna()

svm_values = svm[
    "acc_mean"
].dropna()

all_values = pd.concat(
    [
        hp_values,
        svm_values
    ]
)

y_min = all_values.min()
y_max = all_values.max()

padding = (
    y_max - y_min
) * 0.08

y_min -= padding
y_max += padding


fig.update_yaxes(
    range=[
        y_min,
        y_max
    ]
)


# ============================================================
# Общая подпись X и Y
# ============================================================

fig.update_layout(

    title=(
        "HP-DSC classification accuracy "
        "versus n_splits"
    ),

    template="plotly_white",

    width=1250,
    height=850,

    legend=dict(
        title="Method",

        orientation="h",

        yanchor="bottom",
        y=1.02,

        xanchor="left",
        x=0
    ),

    margin=dict(
        l=100,
        r=40,
        t=100,
        b=70
    )
)


# Убираем индивидуальные подписи X у верхнего ряда,
# чтобы рисунок не был перегружен.

for col in range(1, 4):

    fig.update_xaxes(
        title=None,
        row=1,
        col=col
    )

# Но нижний ряд оставляем с подписью n_splits
for col in range(1, 4):

    fig.update_xaxes(
        title="n_splits",
        row=2,
        col=col
    )


fig.show()


# ============================================================
# Сохранение
# ============================================================


fig.write_image(
    r"D:\Projects\explainable-highdim-svm-recognition\charts\figure_2.svg",
    width=1250,
    height=850
)