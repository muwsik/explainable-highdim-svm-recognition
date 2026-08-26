import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# Файл
# ============================================================

FILE = r"D:\Projects\explainable-highdim-svm-recognition\charts\res002_averaged.xlsx"


# ============================================================
# Параметры эксперимента
# ============================================================

TRAIN_SIZE = 40000

MAX_FEATS = 500
N_EST = 200


# ============================================================
# Читаем ТОЛЬКО Bagging-40k-par2
# ============================================================

bagging = pd.read_excel(
    FILE,
    sheet_name="Bagging-40k-par2"
)


# ============================================================
# Определение регуляризации
#
# penalty='l1' -> L1
# отсутствие penalty -> L2
# ============================================================

def get_regularization(model):

    model = str(model).lower()

    if "penalty='l1'" in model:
        return "L1"

    return "L2"


bagging["regularization"] = (
    bagging["model"]
    .apply(get_regularization)
)


# ============================================================
# Фильтруем единственную комбинацию параметров
# ============================================================

data = bagging[
    (bagging["max_feats"] == MAX_FEATS) &
    (bagging["n_est"] == N_EST)
].copy()


# ============================================================
# Проверка
# ============================================================

print("\nДанные для speedup:")
print(
    data[
        [
            "model",
            "regularization",
            "max_feats",
            "n_est",
            "n_proc",
            "tr_time_mean",
            "tr_time_sd"
        ]
    ]
    .sort_values(
        ["regularization", "n_proc"]
    )
    .to_string(index=False)
)


# ============================================================
# Рисунок
#
#                  L1                     L2
#
#             ┌─────────────┐       ┌─────────────┐
#             │       ideal │       │       ideal │
# speedup     │      /      │       │      /      │
#             │    /        │       │    /        │
#             │  / real     │       │  / real     │
#             └─────────────┘       └─────────────┘
#                   n_proc                n_proc
#
# ============================================================

fig = make_subplots(

    rows=1,
    cols=2,

    subplot_titles=[
        "L1",
        "L2"
    ],

    horizontal_spacing=0.12
)


# ============================================================
# Цвета
# ============================================================

colors = {
    "L1": "#1f77b4",
    "L2": "#ff7f0e"
}


# ============================================================
# Построение
# ============================================================

for col, regularization in enumerate(
    ["L1", "L2"],
    start=1
):

    current = data[
        data["regularization"] == regularization
    ].copy()


    current = current.sort_values(
        "n_proc"
    )


    if len(current) == 0:

        print(
            f"\nНет данных для {regularization}"
        )

        continue


    # --------------------------------------------------------
    # Baseline: время при одном процессе
    # --------------------------------------------------------

    baseline_data = current[
        current["n_proc"] == 1
    ]

    if len(baseline_data) == 0:

        print(
            f"\nНет n_proc=1 для {regularization}"
        )

        continue


    baseline_time = (
        baseline_data[
            "tr_time_mean"
        ].iloc[0]
    )


    # --------------------------------------------------------
    # Speedup
    # --------------------------------------------------------

    current["speedup"] = (
        baseline_time /
        current["tr_time_mean"]
    )


    # --------------------------------------------------------
    # Проверка
    # --------------------------------------------------------

    print(
        f"\n{regularization}: "
        f"baseline = {baseline_time:.4f} s"
    )

    print(
        current[
            [
                "n_proc",
                "tr_time_mean",
                "speedup"
            ]
        ].to_string(index=False)
    )


    # --------------------------------------------------------
    # Реальное ускорение
    # --------------------------------------------------------

    fig.add_trace(

        go.Scatter(

            x=current["n_proc"],

            y=current["speedup"],

            mode="lines",

            name=regularization,

            line=dict(
                color=colors[
                    regularization
                ],
                width=2.5
            ),

            showlegend=False
        ),

        row=1,
        col=col
    )


    # --------------------------------------------------------
    # Идеальное ускорение
    # --------------------------------------------------------

    n_proc = sorted(
        current["n_proc"].unique()
    )


    fig.add_trace(

        go.Scatter(

            x=n_proc,

            y=n_proc,

            mode="lines",

            name="Ideal",

            line=dict(

                color="black",

                width=1.5,

                dash="dash"
            ),

            showlegend=(
                col == 1
            )
        ),

        row=1,
        col=col
    )


    # --------------------------------------------------------
    # Оси
    # --------------------------------------------------------

    fig.update_xaxes(

        title="n_proc",

        tickmode="array",

        tickvals=n_proc,

        row=1,
        col=col
    )


    fig.update_yaxes(

        title="Speedup",

        row=1,
        col=col
    )


# ============================================================
# Общий layout
# ============================================================

fig.update_layout(

    title=(
        "Bagging parallel speedup versus number of processes"
        "<br>"
        f"<sup>"
        f"n = {TRAIN_SIZE // 1000}k, "
        f"max_feats = {MAX_FEATS}, "
        f"n_est = {N_EST}"
        f"</sup>"
    ),

    template="plotly_white",

    width=1200,

    height=550,

    legend=dict(

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

    r"D:\Projects\explainable-highdim-svm-recognition\charts\figure_9.svg",

    width=1200,

    height=550
)


fig.show()