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

# Эти значения соответствуют единственной комбинации
# max_feats × n_est, для которой есть параллельные результаты.
#
# При необходимости проверь их по выводу ниже.

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
# Проверяем структуру данных
# ============================================================

print("\nКолонки:")
print(bagging.columns.tolist())


print("\nВсе комбинации параметров:")
print(
    bagging[
        [
            "model",
            "max_feats",
            "n_est",
            "n_proc"
        ]
    ].drop_duplicates().to_string(index=False)
)


# ============================================================
# Определение регуляризации
#
# penalty='l1' -> L1
# penalty отсутствует -> L2
# ============================================================

def get_regularization(model):

    model = str(model).lower()

    if "penalty='l1'" in model:
        return "L1"

    # В Bagging-40k-par2 отсутствие penalty
    # соответствует L2
    return "L2"


bagging["regularization"] = (
    bagging["model"]
    .apply(get_regularization)
)


# ============================================================
# Проверяем регуляризации
# ============================================================

print("\nРегуляризации:")
print(
    bagging[
        [
            "model",
            "regularization"
        ]
    ].drop_duplicates().to_string(index=False)
)


# ============================================================
# Фильтруем единственную комбинацию
# ============================================================

data = bagging[
    (bagging["max_feats"] == MAX_FEATS) &
    (bagging["n_est"] == N_EST)
].copy()


# ============================================================
# Рисунок
#
#                  L1                     L2
#
#             ┌─────────────┐       ┌─────────────┐
#             │             │       │             │
# time        │      \      │       │      \      │
#             │       \     │       │       \     │
#             │        \    │       │        \    │
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


    # --------------------------------------------------------
    # Проверка
    # --------------------------------------------------------

    print(
        f"\n{regularization}:"
    )

    if len(current) == 0:

        print("Нет данных")

        continue


    print(
        current[
            [
                "n_proc",
                "tr_time_mean",
                "tr_time_sd"
            ]
        ].to_string(index=False)
    )


    # --------------------------------------------------------
    # График
    # --------------------------------------------------------

    fig.add_trace(

        go.Scatter(

            x=current["n_proc"],

            y=current["tr_time_mean"],

            mode="lines",

            name=regularization,

            line=dict(
                color=colors[
                    regularization
                ],
                width=2.5
            ),

            error_y=dict(

                type="data",

                array=current[
                    "tr_time_sd"
                ].fillna(0),

                visible=True,

                color=colors[
                    regularization
                ],

                thickness=1.3,

                width=5
            ),

            showlegend=False
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

        tickvals=sorted(
            current["n_proc"].unique()
        ),

        row=1,
        col=col
    )


    fig.update_yaxes(

        title="Training time, s",

        row=1,
        col=col
    )


# ============================================================
# Общий заголовок
# ============================================================

fig.update_layout(

    title=(
        "Bagging training time versus number of processes"
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

    margin=dict(
        l=90,
        r=50,
        t=110,
        b=80
    )
)


# ============================================================
# Сохранение
# ============================================================


fig.write_image(

    r"D:\Projects\explainable-highdim-svm-recognition\charts\figure_8.svg",

    width=1200,

    height=550
)


fig.show()