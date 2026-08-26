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

n_proc_values = [1, 2, 4, 6, 8, 10]

colorscales = {
    "L1": "Blues",
    "L2": "Oranges"
}


# ============================================================
# Регуляризация
# ============================================================

def get_regularization(model):

    model = str(model).lower()

    if "penalty='l1'" in model or 'penalty="l1"' in model:
        return "L1"

    return "L2"


# ============================================================
# Чтение HP-DSC
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
# Рисунок 2 × 3
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


        # ----------------------------------------------------
        # Данные
        # ----------------------------------------------------

        data = hp[
            (hp["regularization"] == regularization) &
            (hp["train_size"] == size) &
            (hp["n_proc"].isin(n_proc_values))
        ].copy()


        # ----------------------------------------------------
        # Равномерные координаты n_proc
        # ----------------------------------------------------

        n_proc_index = {
            value: i
            for i, value in enumerate(n_proc_values)
        }

        data["x_index"] = data["n_proc"].map(
            n_proc_index
        )


        # ----------------------------------------------------
        # Равномерные координаты n_splits
        # ----------------------------------------------------

        n_splits_values = sorted(
            data["n_splits"].dropna().unique()
        )

        n_splits_index = {
            value: i
            for i, value in enumerate(n_splits_values)
        }

        data["y_index"] = data["n_splits"].map(
            n_splits_index
        )


        # ----------------------------------------------------
        # Таблица времени
        # ----------------------------------------------------

        time_table = data.pivot(
            index="y_index",
            columns="x_index",
            values="tr_time_mean"
        )

        time_table = time_table.reindex(
            index=range(len(n_splits_values)),
            columns=range(len(n_proc_values))
        )


        # ----------------------------------------------------
        # Время при n_proc = 1
        #
        # Для каждого n_splits это baseline.
        # ----------------------------------------------------

        baseline = time_table[0]


        # ----------------------------------------------------
        # Speedup
        # ----------------------------------------------------

        speedup = time_table.divide(
            baseline,
            axis=0
        )


        # ----------------------------------------------------
        # Текст внутри ячеек
        # ----------------------------------------------------

        text = []

        for y in range(
            len(n_splits_values)
        ):

            row_text = []

            for x in range(
                len(n_proc_values)
            ):

                value = speedup.loc[y, x]

                if pd.isna(value):

                    row_text.append("")

                else:

                    row_text.append(
                        f"{value:.2f}"
                    )

            text.append(row_text)


        # ----------------------------------------------------
        # Собственный диапазон цветовой шкалы
        # ----------------------------------------------------

        valid_values = speedup.values[
            ~pd.isna(speedup.values)
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
                    range(len(n_proc_values))
                ),

                y=list(
                    range(len(n_splits_values))
                ),

                z=speedup.values,

                coloraxis=coloraxis_name,

                text=text,

                texttemplate="%{text}",

                textfont=dict(
                    size=10
                ),

                hoverinfo="skip",

                xgap=0,
                ygap=0,

                showscale=True
            ),

            row=row,
            col=col
        )


        # ----------------------------------------------------
        # Colorbar
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
                            text="Speedup"
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
        # X
        # ----------------------------------------------------

        fig.update_xaxes(

            tickmode="array",

            tickvals=list(
                range(len(n_proc_values))
            ),

            ticktext=[
                str(x)
                for x in n_proc_values
            ],

            title="n_proc",

            range=[
                -0.5,
                len(n_proc_values) - 0.5
            ],

            row=row,
            col=col
        )


        # ----------------------------------------------------
        # Y
        # ----------------------------------------------------

        fig.update_yaxes(

            tickmode="array",

            tickvals=list(
                range(len(n_splits_values))
            ),

            ticktext=[
                str(x)
                for x in n_splits_values
            ],

            title=(
                "n_splits"
                if col == 1
                else None
            ),

            range=[
                len(n_splits_values) - 0.5,
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

        fig.update_xaxes(
            scaleanchor=f"y{subplot_number}",
            scaleratio=1,
            constrain="domain",
            row=row,
            col=col
        )


# ============================================================
# L1 / L2
# ============================================================

fig.add_annotation(
    text="<b>L1</b>",
    x=-0.055,
    y=0.75,
    xref="paper",
    yref="paper",
    textangle=-90,
    showarrow=False,
    font=dict(size=16)
)

fig.add_annotation(
    text="<b>L2</b>",
    x=-0.055,
    y=0.25,
    xref="paper",
    yref="paper",
    textangle=-90,
    showarrow=False,
    font=dict(size=16)
)


# ============================================================
# Общий layout
# ============================================================

fig.update_layout(

    title=(
        "HP-DSC parallel speedup "
        "as a function of n_splits and n_proc"
    ),

    template="plotly_white",

    width=1500,
    height=900,

    margin=dict(
        l=100,
        r=100,
        t=100,
        b=90
    )
)


# ============================================================
# Сохранение
# ============================================================


fig.write_image(
    r"D:\Projects\explainable-highdim-svm-recognition\charts\figure_5.svg",
    width=1500,
    height=900
)


fig.show()