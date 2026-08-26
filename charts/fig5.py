import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


FILE = r"D:\Projects\explainable-highdim-svm-recognition\charts\res002_averaged.xlsx"


sizes = [20000, 30000, 40000]
regularizations = ["L1", "L2"]

n_proc_values = [1, 2, 4, 6, 8, 10]

colorscales = {
    "L1": "Blues",
    "L2": "Oranges"
}


def get_regularization(model):

    model = str(model).lower()

    if "penalty='l1'" in model or 'penalty="l1"' in model:
        return "L1"

    return "L2"


# ============================================================
# Читаем HP-DSC
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
# Проверяем данные перед расчётом
# ============================================================

print("\nПРОВЕРКА BASELINE n_proc=1")
print("=" * 60)

for reg in regularizations:

    for size in sizes:

        data = hp[
            (hp["regularization"] == reg) &
            (hp["train_size"] == size)
        ].copy()

        print(f"\n{reg}, {size}")

        for n_splits in sorted(
            data["n_splits"].dropna().unique()
        ):

            row = data[
                (data["n_splits"] == n_splits) &
                (data["n_proc"] == 1)
            ]

            if len(row) == 0:

                print(
                    f"n_splits={n_splits}: "
                    "NO n_proc=1 BASELINE"
                )

            elif len(row) > 1:

                print(
                    f"n_splits={n_splits}: "
                    f"WARNING: {len(row)} baseline rows"
                )

            else:

                print(
                    f"n_splits={n_splits}: "
                    f"T1 = {row['tr_time_mean'].iloc[0]:.4f} s"
                )


# ============================================================
# Рисунок
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


coloraxis_number = 0


for row, reg in enumerate(
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
            (hp["regularization"] == reg) &
            (hp["train_size"] == size) &
            (hp["n_proc"].isin(n_proc_values))
        ].copy()


        # ----------------------------------------------------
        # Получаем baseline T(n_proc=1)
        # ----------------------------------------------------

        baseline = (
            data[
                data["n_proc"] == 1
            ]
            .set_index("n_splits")[
                "tr_time_mean"
            ]
        )


        # ----------------------------------------------------
        # Матрица времени
        # ----------------------------------------------------

        time_table = data.pivot(
            index="n_splits",
            columns="n_proc",
            values="tr_time_mean"
        )


        time_table = time_table.reindex(
            columns=n_proc_values
        )


        # ----------------------------------------------------
        # Speedup
        # ----------------------------------------------------

        speedup = pd.DataFrame(
            index=time_table.index,
            columns=time_table.columns,
            dtype=float
        )


        for n_splits in speedup.index:

            # Нет последовательного baseline
            if n_splits not in baseline.index:
                continue

            t1 = baseline.loc[n_splits]

            # Защита от некорректного времени
            if pd.isna(t1) or t1 <= 0:
                continue

            for n_proc in n_proc_values:

                tp = time_table.loc[
                    n_splits,
                    n_proc
                ]

                if pd.isna(tp) or tp <= 0:
                    continue

                speedup.loc[
                    n_splits,
                    n_proc
                ] = t1 / tp


        # ----------------------------------------------------
        # КРИТИЧЕСКАЯ ПРОВЕРКА
        # ----------------------------------------------------

        print(
            f"\n{reg}, {size}: speedup"
        )

        print(
            speedup.round(2)
        )


        # ----------------------------------------------------
        # Текст
        # ----------------------------------------------------

        text = []

        for n_splits in speedup.index:

            row_text = []

            for n_proc in n_proc_values:

                value = speedup.loc[
                    n_splits,
                    n_proc
                ]

                if pd.isna(value):

                    row_text.append("")

                else:

                    row_text.append(
                        f"{value:.2f}"
                    )

            text.append(row_text)


        # ----------------------------------------------------
        # Цветовая шкала
        # ----------------------------------------------------

        valid_values = speedup.values[
            ~pd.isna(speedup.values)
        ]

        if len(valid_values) > 0:

            zmin = valid_values.min()
            zmax = valid_values.max()

        else:

            zmin = 1
            zmax = 2


        # ----------------------------------------------------
        # Heatmap
        # ----------------------------------------------------

        fig.add_trace(

            go.Heatmap(

                x=list(
                    range(len(n_proc_values))
                ),

                y=list(
                    range(len(speedup.index))
                ),

                z=speedup.values,

                coloraxis=coloraxis_name,

                text=text,

                texttemplate="%{text}",

                textfont=dict(
                    size=10
                ),

                hoverinfo="skip"
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

                    colorscale=colorscales[reg],

                    cmin=zmin,
                    cmax=zmax,

                    colorbar=dict(

                        title="Speedup",

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

        n_splits_values = list(
            speedup.index
        )

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
# Layout
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