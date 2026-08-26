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

regularizations = ["L1", "L2"]

max_feats_values = [
    10, 50, 100, 200, 250, 300, 500
]

n_est_values = [
    10, 20, 30, 50, 100, 200
]

colorscales = {
    "L1": "Blues",
    "L2": "Oranges"
}


# ============================================================
# Определение регуляризации
# ============================================================

def get_regularization(model):

    model = str(model).lower()

    if "penalty='l1'" in model or 'penalty="l1"' in model:
        return "L1"

    return "L2"


# ============================================================
# Чтение Bagging
# ============================================================

bagging_sheets = {
    20000: "Bagging-20k",
    30000: "Bagging-30k",
    40000: "Bagging-40k"
}

bagging = []

for size, sheet in bagging_sheets.items():

    df = pd.read_excel(
        FILE,
        sheet_name=sheet
    )

    df["train_size"] = size

    df["regularization"] = df["model"].apply(
        get_regularization
    )

    bagging.append(df)


bagging = pd.concat(
    bagging,
    ignore_index=True
)


# ============================================================
# Рисунок 2 × 3
#
#                 20k          30k          40k
#
#           ┌────────────┬────────────┬────────────┐
#       L1  │            │            │            │
#           │  heatmap   │  heatmap   │  heatmap   │
#           ├────────────┼────────────┼────────────┤
#       L2  │            │            │            │
#           │  heatmap   │  heatmap   │  heatmap   │
#           └────────────┴────────────┴────────────┘
#
# X = n_est
# Y = max_feats
# Z = training time
# ============================================================

fig = make_subplots(

    rows=2,
    cols=3,

    subplot_titles=[
        "20k", "30k", "40k",
        "20k", "30k", "40k"
    ],

    horizontal_spacing=0.09,
    vertical_spacing=0.15
)


# ============================================================
# Построение heatmap
# ============================================================

coloraxis_number = 0


for row, regularization in enumerate(
    regularizations,
    start=1
):

    for col, size in enumerate(
        sizes,
        start=1
    ):

        coloraxis_number += 1

        coloraxis_name = (
            f"coloraxis{coloraxis_number}"
        )

        subplot_number = (
            (row - 1) * 3 + col
        )


        # ----------------------------------------------------
        # Данные конкретной панели
        # ----------------------------------------------------

        data = bagging[
            (bagging["regularization"] == regularization) &
            (bagging["train_size"] == size)
        ].copy()


        # ----------------------------------------------------
        # Равномерные координаты X
        # ----------------------------------------------------

        n_est_index = {
            value: i
            for i, value in enumerate(n_est_values)
        }

        data["x_index"] = data["n_est"].map(
            n_est_index
        )


        # ----------------------------------------------------
        # Равномерные координаты Y
        # ----------------------------------------------------

        max_feats_index = {
            value: i
            for i, value in enumerate(max_feats_values)
        }

        data["y_index"] = data["max_feats"].map(
            max_feats_index
        )


        # ----------------------------------------------------
        # Матрица среднего времени
        # ----------------------------------------------------

        heatmap = data.pivot(
            index="y_index",
            columns="x_index",
            values="tr_time_mean"
        )

        heatmap = heatmap.reindex(
            index=range(len(max_feats_values)),
            columns=range(len(n_est_values))
        )


        # ----------------------------------------------------
        # Матрица СКО
        # ----------------------------------------------------

        sd = data.pivot(
            index="y_index",
            columns="x_index",
            values="tr_time_sd"
        )

        sd = sd.reindex(
            index=heatmap.index,
            columns=heatmap.columns
        )


        # ----------------------------------------------------
        # Текст внутри ячеек
        # ----------------------------------------------------

        text = []

        for y in range(
            len(max_feats_values)
        ):

            row_text = []

            for x in range(
                len(n_est_values)
            ):

                mean_value = heatmap.loc[y, x]
                sd_value = sd.loc[y, x]

                if pd.isna(mean_value):

                    row_text.append("")

                else:

                    if pd.isna(sd_value):
                        sd_value = 0

                    row_text.append(
                        f"{mean_value:.2f}"
                        f"<br>±{sd_value:.2f}"
                    )

            text.append(row_text)


        # ----------------------------------------------------
        # Диапазон собственной цветовой шкалы
        # ----------------------------------------------------

        valid_values = heatmap.values[
            ~pd.isna(heatmap.values)
        ]

        if len(valid_values) > 0:

            zmin = valid_values.min()
            zmax = valid_values.max()

        else:

            zmin = 0
            zmax = 1


        # ----------------------------------------------------
        # Heatmap
        # ----------------------------------------------------

        fig.add_trace(

            go.Heatmap(

                x=list(
                    range(len(n_est_values))
                ),

                y=list(
                    range(len(max_feats_values))
                ),

                z=heatmap.values,

                coloraxis=coloraxis_name,

                text=text,

                texttemplate="%{text}",

                textfont=dict(
                    size=9
                ),

                hoverinfo="skip",

                xgap=0,
                ygap=0
            ),

            row=row,
            col=col
        )


        # ----------------------------------------------------
        # Собственная colorbar
        # ----------------------------------------------------

        colorbar_x = {
            1: 0.285,
            2: 0.625,
            3: 0.965
        }[col]

        colorbar_y = {
            1: 0.75,
            2: 0.25
        }[row]


        fig.update_layout(

            **{
                coloraxis_name: dict(

                    colorscale=colorscales[
                        regularization
                    ],

                    cmin=zmin,
                    cmax=zmax,

                    colorbar=dict(

                        title=dict(
                            text="Training time, s"
                        ),

                        x=colorbar_x,
                        y=colorbar_y,

                        len=0.30,

                        thickness=12
                    )
                )
            }
        )


        # ----------------------------------------------------
        # Ось X
        # ----------------------------------------------------

        fig.update_xaxes(

            tickmode="array",

            tickvals=list(
                range(len(n_est_values))
            ),

            ticktext=[
                str(x)
                for x in n_est_values
            ],

            title="n_est",

            range=[
                -0.5,
                len(n_est_values) - 0.5
            ],

            row=row,
            col=col
        )


        # ----------------------------------------------------
        # Ось Y
        # ----------------------------------------------------

        fig.update_yaxes(

            tickmode="array",

            tickvals=list(
                range(len(max_feats_values))
            ),

            ticktext=[
                str(x)
                for x in max_feats_values
            ],

            title=(
                "max_feats"
                if col == 1
                else None
            ),

            range=[
                len(max_feats_values) - 0.5,
                -0.5
            ],

            row=row,
            col=col
        )


# ============================================================
# Квадратные ячейки
# ============================================================

for row in range(1, 3):

    for col in range(1, 4):

        subplot_number = (
            (row - 1) * 3 + col
        )

        if subplot_number == 1:
            y_axis = "y"
        else:
            y_axis = f"y{subplot_number}"

        fig.update_xaxes(

            scaleanchor=y_axis,

            scaleratio=1,

            constrain="domain",

            row=row,
            col=col
        )


# ============================================================
# Подписи L1 / L2
# ============================================================

fig.add_annotation(

    text="<b>L1</b>",

    x=-0.055,
    y=0.75,

    xref="paper",
    yref="paper",

    textangle=-90,

    showarrow=False,

    font=dict(
        size=16
    )
)


fig.add_annotation(

    text="<b>L2</b>",

    x=-0.055,
    y=0.25,

    xref="paper",
    yref="paper",

    textangle=-90,

    showarrow=False,

    font=dict(
        size=16
    )
)


# ============================================================
# Общий layout
# ============================================================

fig.update_layout(

    title=(
        "Bagging training time "
        "as a function of n_est and max_feats"
    ),

    template="plotly_white",

    width=1500,
    height=950,

    margin=dict(
        l=105,
        r=100,
        t=110,
        b=90
    )
)


# ============================================================
# Сохранение
# ============================================================

fig.write_image(

    r"D:\Projects\explainable-highdim-svm-recognition\charts\figure_7.svg",

    width=1500,
    height=950
)


# ============================================================
# Показ
# ============================================================

fig.show()