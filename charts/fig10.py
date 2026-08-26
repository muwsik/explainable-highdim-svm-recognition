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

TRAIN_SIZE = 40000
MAX_N_PROC = 10

REGULARIZATIONS = ["L1", "L2"]


# ============================================================
# Цвета
# ============================================================

METHOD_COLORS = {
    "SVM": "#1f77b4",
    "HP-DSC": "#ff7f0e",
    "Bagging": "#2ca02c"
}


# ============================================================
# Регуляризация
#
# penalty='l1' -> L1
# отсутствие penalty -> L2
# ============================================================

def get_regularization(model):

    model = str(model).lower()

    if "penalty='l1'" in model:
        return "L1"

    if 'penalty="l1"' in model:
        return "L1"

    return "L2"


# ============================================================
# SVM
# ============================================================

svm = pd.read_excel(
    FILE,
    sheet_name="SVM"
)

svm["regularization"] = (
    svm["model"]
    .apply(get_regularization)
)

svm = svm[
    svm["nTr_obj"] == TRAIN_SIZE
].copy()


# ============================================================
# HP-DSC
# ============================================================

hp = pd.read_excel(
    FILE,
    sheet_name="HP-DSC-40k"
)

hp["regularization"] = (
    hp["model"]
    .apply(get_regularization)
)

hp = hp[
    hp["n_proc"] == MAX_N_PROC
].copy()


# ============================================================
# Bagging
#
# Используем только Bagging-40k-par2
# ============================================================

bagging = pd.read_excel(
    FILE,
    sheet_name="Bagging-40k-par2"
)

bagging["regularization"] = (
    bagging["model"]
    .apply(get_regularization)
)

bagging = bagging[
    bagging["n_proc"] == MAX_N_PROC
].copy()


# ============================================================
# Проверка
# ============================================================

print("\n================ SVM ================")

print(
    svm[
        [
            "regularization",
            "acc_mean",
            "acc_sd",
            "tr_time_mean",
            "tr_time_sd"
        ]
    ]
    .sort_values("regularization")
    .to_string(index=False)
)


print("\n================ HP-DSC ================")

print(
    hp[
        [
            "regularization",
            "n_splits",
            "n_proc",
            "acc_tst_mean",
            "acc_tst_sd",
            "tr_time_mean",
            "tr_time_sd"
        ]
    ]
    .sort_values(
        ["regularization", "n_splits"]
    )
    .to_string(index=False)
)


print("\n================ Bagging ================")

print(
    bagging[
        [
            "regularization",
            "max_feats",
            "n_est",
            "n_proc",
            "acc_mean",
            "acc_sd",
            "tr_time_mean",
            "tr_time_sd"
        ]
    ]
    .sort_values("regularization")
    .to_string(index=False)
)


# ============================================================
# Рисунок
# ============================================================

fig = make_subplots(

    rows=1,
    cols=2,

    subplot_titles=[
        "L1",
        "L2"
    ],

    horizontal_spacing=0.10
)


# ============================================================
# Построение
# ============================================================

for col, regularization in enumerate(
    REGULARIZATIONS,
    start=1
):

    # ========================================================
    # SVM baseline
    # ========================================================

    svm_current = svm[
        svm["regularization"] == regularization
    ].copy()


    if len(svm_current) > 0:

        svm_accuracy = (
            svm_current[
                "acc_mean"
            ].iloc[0]
        )

        svm_time = (
            svm_current[
                "tr_time_mean"
            ].iloc[0]
        )


        # ----------------------------------------------------
        # Диапазон X
        #
        # Линия проходит по всей области графика.
        # ----------------------------------------------------

        hp_current = hp[
            hp["regularization"] == regularization
        ]

        bagging_current = bagging[
            bagging["regularization"] == regularization
        ]


        all_times = pd.concat(
            [
                hp_current["tr_time_mean"],
                bagging_current["tr_time_mean"]
            ]
        ).dropna()


        if len(all_times) > 0:

            xmin = all_times.min()
            xmax = all_times.max()

            # Небольшой запас слева и справа
            padding = (
                xmax - xmin
            ) * 0.08

            if padding == 0:
                padding = 1

            line_x = [
                max(0, xmin - padding),
                xmax + padding
            ]

        else:

            line_x = [
                0,
                svm_time
            ]


        # ----------------------------------------------------
        # Горизонтальная линия SVM
        # ----------------------------------------------------

        fig.add_trace(

            go.Scatter(

                x=line_x,

                y=[
                    svm_accuracy,
                    svm_accuracy
                ],

                mode="lines",

                name="SVM",

                legendgroup="SVM",

                showlegend=(
                    col == 1
                ),

                line=dict(

                    color=METHOD_COLORS["SVM"],

                    width=2.5,

                    dash="dash"
                ),

                hoverinfo="skip"
            ),

            row=1,
            col=col
        )


        # ----------------------------------------------------
        # Подпись SVM
        # ----------------------------------------------------

        fig.add_annotation(

            text=(
                f"<b>SVM</b>: "
                f"accuracy = {svm_accuracy:.4f}"
                f"<br>"
                f"time = {svm_time:.1f} s"
            ),

            x=0.98,

            y=svm_accuracy,

            xref=(
                "x domain"
                if col == 1
                else f"x{col} domain"
            ),

            yref=(
                "y"
                if col == 1
                else f"y{col}"
            ),

            xanchor="right",

            yanchor="bottom",

            showarrow=False,

            font=dict(
                size=11
            )
        )


    # ========================================================
    # HP-DSC
    # ========================================================

    hp_current = hp[
        hp["regularization"] == regularization
    ].copy()


    if len(hp_current) > 0:

        fig.add_trace(

            go.Scatter(

                x=hp_current[
                    "tr_time_mean"
                ],

                y=hp_current[
                    "acc_tst_mean"
                ],

                mode="markers",

                name="HP-DSC",

                legendgroup="HP-DSC",

                showlegend=(
                    col == 1
                ),

                marker=dict(

                    color=METHOD_COLORS["HP-DSC"],

                    size=9,

                    symbol="circle"
                ),

                error_x=dict(

                    type="data",

                    array=hp_current[
                        "tr_time_sd"
                    ].fillna(0),

                    visible=True,

                    thickness=1.2,

                    width=4
                ),

                error_y=dict(

                    type="data",

                    array=hp_current[
                        "acc_tst_sd"
                    ].fillna(0),

                    visible=True,

                    thickness=1.2,

                    width=4
                ),

                hoverinfo="skip"
            ),

            row=1,
            col=col
        )


    # ========================================================
    # Bagging
    # ========================================================

    bagging_current = bagging[
        bagging["regularization"] == regularization
    ].copy()


    if len(bagging_current) > 0:

        fig.add_trace(

            go.Scatter(

                x=bagging_current[
                    "tr_time_mean"
                ],

                y=bagging_current[
                    "acc_mean"
                ],

                mode="markers",

                name="Bagging",

                legendgroup="Bagging",

                showlegend=(
                    col == 1
                ),

                marker=dict(

                    color=METHOD_COLORS["Bagging"],

                    size=11,

                    symbol="circle"
                ),

                error_x=dict(

                    type="data",

                    array=bagging_current[
                        "tr_time_sd"
                    ].fillna(0),

                    visible=True,

                    thickness=1.2,

                    width=4
                ),

                error_y=dict(

                    type="data",

                    array=bagging_current[
                        "acc_sd"
                    ].fillna(0),

                    visible=True,

                    thickness=1.2,

                    width=4
                ),

                hoverinfo="skip"
            ),

            row=1,
            col=col
        )


    # ========================================================
    # Оси
    # ========================================================

    fig.update_xaxes(

        title="Training time, s",

        row=1,
        col=col
    )

    fig.update_yaxes(

        title="Test accuracy",

        row=1,
        col=col
    )


# ============================================================
# Layout
# ============================================================

fig.update_layout(

    title=(
        "Accuracy versus training time"
        "<br>"
        f"<sup>"
        f"training size = {TRAIN_SIZE // 1000}k; "
        f"parallel methods: n_proc = {MAX_N_PROC}"
        f"</sup>"
    ),

    template="plotly_white",

    width=1200,

    height=650,

    legend=dict(

        title="Method",

        orientation="h",

        yanchor="bottom",

        y=1.02,

        xanchor="left",

        x=0
    ),

    margin=dict(

        l=90,
        r=50,
        t=120,
        b=80
    )
)


# ============================================================
# Сохранение
# ============================================================


fig.write_image(

    r"D:\Projects\explainable-highdim-svm-recognition\charts\figure_1.svg",

    width=1200,
    height=650
)


fig.show()