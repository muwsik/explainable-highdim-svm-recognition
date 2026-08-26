import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------------------------------
# 1. Load aggregated sheet
# -------------------------------------------------
fileID = "s3.1"

file = r"D:\Cloud\SVM\charts\s3.1_synthetic6-10k-f1000-i500-r250-l250.npz.xlsx"

df = pd.read_excel(
    file,
    sheet_name="aggregated"
)

# -------------------------------------------------
# 2. Parameters
# -------------------------------------------------
metric_mean = "Acc(test)_mean"
metric_std = "Acc(test)_cv_std_mean"

model_name = "Comb-LSVC-l1"

train_sizes = [50, 100, 150, 200, 250, 500]

# -------------------------------------------------
# 3. Filter dataframe
# -------------------------------------------------
df_line = df[
    df["model"].isin([
        "Comb-LSVC-l1",
        "Comb-LSVC-l2"
    ])
]

df = df[df["model"] == model_name]

df = df.sort_values(["train-size", "C", "splits"])

# -------------------------------------------------
# 4. Global color scale
# -------------------------------------------------
vmin = df[metric_mean].min() * 100
vmax = df[metric_mean].max() * 100

# -------------------------------------------------
# 5. Heatmaps
# -------------------------------------------------
fig = make_subplots(
    rows=1,
    cols=len(train_sizes),
    subplot_titles=[
        f"Train size = {ts}"
        for ts in train_sizes
    ],

    horizontal_spacing=0.01
)

for idx, train_size in enumerate(train_sizes, start=1):

    temp = df[df["train-size"] == train_size]

    pivot_mean = temp.pivot(
        index="splits",
        columns="C",
        values=metric_mean
    ) * 100

    pivot_std = temp.pivot(
        index="splits",
        columns="C",
        values=metric_std
    ) * 100

    # annotations
    text = []

    for i in range(pivot_mean.shape[0]):

        row = []

        for j in range(pivot_mean.shape[1]):

            mean_val = pivot_mean.iloc[i, j]
            std_val = pivot_std.iloc[i, j]

            row.append(
                f"{mean_val:.1f}<br>±{std_val:.1f}"
            )

        text.append(row)

    heatmap = go.Heatmap(

        z=pivot_mean.values,

        x=pivot_mean.columns.astype(str),
        y=pivot_mean.index.astype(str),

        text=text,
        texttemplate="%{text}",

        colorscale="Viridis",

        zmin=vmin,
        zmax=vmax,

        showscale=False,

        hovertemplate=
            "C=%{x}<br>" +
            "Splits=%{y}<br>" +
            "Accuracy=%{z:.2f}%<extra></extra>"
    )

    fig.add_trace(
        heatmap,
        row=1,
        col=idx
    )

    fig.update_xaxes(        
        constrain="domain",
        range=[-0.5, len(pivot_mean.columns) - 0.5],
        row=1,
        col=idx
    )

    fig.update_yaxes(
        constrain="domain",
        range=[len(pivot_mean.index) - 0.5, -0.5],
        row=1,
        col=idx
    )

# -------------------------------------------------
# 6. Layout
# -------------------------------------------------
fig.update_layout(

    title=f"{model_name} — Accuracy (%)",

    height=500,
    width=300 * len(train_sizes),

    template="plotly_white"
)

# квадратные клетки
for i in range(1, len(train_sizes) + 1):

    fig.update_yaxes(
        scaleanchor=f"x{i}",
        scaleratio=1,
        row=1,
        col=i
    )

# -------------------------------------------------
# 7. Save
# -------------------------------------------------
fig.write_html(
    rf"D:\Cloud\SVM\charts\\{fileID}_heatmap_{model_name}.html"
)

print("Heatmap saved")

# -------------------------------------------------
# 8. Line plot
# -------------------------------------------------
train_colors = {
    50: "blue",
    100: "orange",
    150: "green",
    200: "red",
    250: "purple",
    500: "brown",
    1000: "darkmagenta"
}

selected_C = 1

temp_df = df_line[
    df_line["C"] == selected_C
].copy()

temp_df["penalty"] = temp_df["model"].apply(
    lambda x: "l1" if "l1" in x.lower() else "l2"
)

fig_line = go.Figure()

all_train_sizes = sorted(
    temp_df["train-size"].unique()
)

for penalty in ["l1", "l2"]:

    for train_size in all_train_sizes:

        temp = temp_df[
            (temp_df["penalty"] == penalty) &
            (temp_df["train-size"] == train_size)
        ].sort_values("splits")

        if penalty == "l1":
            dash = "dash"
            marker_symbol = "square"
        else:
            dash = "solid"
            marker_symbol = "circle"

        fig_line.add_trace(

            go.Scatter(

                x=temp["splits"],
                y=temp[metric_mean] * 100,

                mode="lines+markers",

                line=dict(
                    color=train_colors[train_size],
                    dash=dash,
                    width=2
                ),

                marker=dict(
                    symbol=marker_symbol,
                    size=6
                ),

                name=f"{penalty} train={train_size}",

                hovertemplate=
                    "Splits=%{x}<br>" +
                    "Accuracy=%{y:.2f}%<extra></extra>"
            )
        )

# -------------------------------------------------
# 9. Layout
# -------------------------------------------------
fig_line.update_layout(

    title=f"Accuracy vs Splits (C={selected_C})",

    xaxis_title="Splits",
    yaxis_title="Accuracy (%)",

    template="plotly_white",

    width=1000,
    height=600
)

# -------------------------------------------------
# 10. Save
# -------------------------------------------------
fig_line.write_html(
    rf"D:\Cloud\SVM\charts\{fileID}_line_C{selected_C}_splits.html"
)

print("Line plot saved")

# -------------------------------------------------
# TRAIN TIME + ITERATIONS + ACCURACY HEATMAPS
# -------------------------------------------------

train_size_fixed = 2500

model_name = "Comb-LSVC-l1"

# -------------------------------------------------
# FILTER
# -------------------------------------------------
df_time = df_line[
    (df_line["train-size"] == train_size_fixed) &
    (df_line["model"] == model_name)
].copy()

# -------------------------------------------------
# GLOBAL RANGES
# -------------------------------------------------
time_vmin = df_time["time(train)_mean"].min()
time_vmax = df_time["time(train)_mean"].max()

iter_vmin = df_time["n_iter_mean"].min()
iter_vmax = df_time["n_iter_mean"].max()

acc_vmin = df_time["Acc(test)_mean"].min() * 100
acc_vmax = df_time["Acc(test)_mean"].max() * 100

# -------------------------------------------------
# SUBPLOTS
# -------------------------------------------------
fig_time = make_subplots(

    rows=1,
    cols=3,

    subplot_titles=[
        "Train time",
        "Iterations",
        "Accuracy"
    ],

    horizontal_spacing=0.05
)

# -------------------------------------------------
# PIVOTS
# -------------------------------------------------
pivot_time_mean = df_time.pivot(
    index="splits",
    columns="processes",
    values="time(train)_mean"
)

pivot_time_std = df_time.pivot(
    index="splits",
    columns="processes",
    values="time(train)_std"
)

pivot_iter = df_time.pivot(
    index="splits",
    columns="processes",
    values="n_iter_mean"
)

pivot_acc_mean = df_time.pivot(
    index="splits",
    columns="processes",
    values="Acc(test)_mean"
) * 100

pivot_acc_std = df_time.pivot(
    index="splits",
    columns="processes",
    values="Acc(test)_std"
) * 100

# -------------------------------------------------
# TIME TEXT
# -------------------------------------------------
text_time = []

for i in range(pivot_time_mean.shape[0]):

    row = []

    for j in range(pivot_time_mean.shape[1]):

        mean_val = pivot_time_mean.iloc[i, j]
        std_val = pivot_time_std.iloc[i, j]

        row.append(
            f"{mean_val:.1f}<br>±{std_val:.1f}"
        )

    text_time.append(row)

# -------------------------------------------------
# TIME HEATMAP
# -------------------------------------------------
heatmap_time = go.Heatmap(

    z=pivot_time_mean.values,

    x=pivot_time_mean.columns.astype(str),
    y=pivot_time_mean.index.astype(str),

    text=text_time,
    texttemplate="%{text}",

    colorscale="Viridis",

    zmin=time_vmin,
    zmax=time_vmax,

    showscale=False,

    hovertemplate=
        "Processes=%{x}<br>" +
        "Splits=%{y}<br>" +
        "Time=%{z:.3f}s<extra></extra>"
)

fig_time.add_trace(
    heatmap_time,
    row=1,
    col=1
)

# -------------------------------------------------
# ITER TEXT
# -------------------------------------------------
text_iter = []

for i in range(pivot_iter.shape[0]):

    row = []

    for j in range(pivot_iter.shape[1]):

        val = pivot_iter.iloc[i, j]

        row.append(
            f"{val:.0f}"
        )

    text_iter.append(row)

# -------------------------------------------------
# ITER HEATMAP
# -------------------------------------------------
heatmap_iter = go.Heatmap(

    z=pivot_iter.values,

    x=pivot_iter.columns.astype(str),
    y=pivot_iter.index.astype(str),

    text=text_iter,
    texttemplate="%{text}",

    colorscale="Viridis",

    zmin=iter_vmin,
    zmax=iter_vmax,

    showscale=False,

    hovertemplate=
        "Processes=%{x}<br>" +
        "Splits=%{y}<br>" +
        "Iterations=%{z:.0f}<extra></extra>"
)

fig_time.add_trace(
    heatmap_iter,
    row=1,
    col=2
)

# -------------------------------------------------
# ACCURACY TEXT
# -------------------------------------------------
text_acc = []

for i in range(pivot_acc_mean.shape[0]):

    row = []

    for j in range(pivot_acc_mean.shape[1]):

        mean_val = pivot_acc_mean.iloc[i, j]
        std_val = pivot_acc_std.iloc[i, j]

        row.append(
            f"{mean_val:.1f}<br>±{std_val:.1f}"
        )

    text_acc.append(row)

# -------------------------------------------------
# ACCURACY HEATMAP
# -------------------------------------------------
heatmap_acc = go.Heatmap(

    z=pivot_acc_mean.values,

    x=pivot_acc_mean.columns.astype(str),
    y=pivot_acc_mean.index.astype(str),

    text=text_acc,
    texttemplate="%{text}",

    colorscale="Viridis",

    zmin=acc_vmin,
    zmax=acc_vmax,

    showscale=False,

    hovertemplate=
        "Processes=%{x}<br>" +
        "Splits=%{y}<br>" +
        "Accuracy=%{z:.2f}%<extra></extra>"
)

fig_time.add_trace(
    heatmap_acc,
    row=1,
    col=3
)

# -------------------------------------------------
# AXES
# -------------------------------------------------
for col in [1, 2, 3]:

    fig_time.update_xaxes(
        title_text="Processes",
        constrain="domain",

        row=1,
        col=col
    )

    fig_time.update_yaxes(
        title_text="Splits",
        constrain="domain",

        autorange="reversed",

        row=1,
        col=col
    )

# -------------------------------------------------
# LAYOUT
# -------------------------------------------------
fig_time.update_layout(

    title= "25k-f5000-i4000-r500-l500 "
        f"{model_name} "
        f"(train-size={train_size_fixed})",

    width=1200,
    height=400,

    template="plotly_white",

    margin=dict(
        l=10,
        r=10,
        t=50,
        b=10
    )
)

# -------------------------------------------------
# SAVE
# -------------------------------------------------
fig_time.write_html(
    rf"D:\Cloud\SVM\charts\{fileID}_time_iter_acc_heatmap.html"
)

print("Train time + iterations + accuracy heatmaps saved")