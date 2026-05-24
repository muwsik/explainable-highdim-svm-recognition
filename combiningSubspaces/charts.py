import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------
# 1. Load aggregated sheet
# -------------------------------------------------
fileID = "s2.3"

file = r"D:\Muws\svm\s2.3_synthetic9-10k-f1000-i750-r0-l250.npz.xlsx"

df = pd.read_excel(
    file,
    sheet_name="aggregated"
)

# -------------------------------------------------
# 2. Parameters
# -------------------------------------------------
metric_mean = "Acc(test)_mean"
metric_std = "Acc(test)_cv_std_mean"
# metric_std = "Acc(test)_std"

model_name = "Comb-LSVC-l2"

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
# 4. Figure
# -------------------------------------------------
fig, axes = plt.subplots(
    1,
    len(train_sizes),
    figsize=(5 * len(train_sizes), 5)
)

plt.subplots_adjust(
    wspace=0.3
)

# общий диапазон цветов
vmin = df[metric_mean].min() * 100
vmax = df[metric_mean].max() * 100

# -------------------------------------------------
# 5. Heatmaps
# -------------------------------------------------
for ax, train_size in zip(axes, train_sizes):

    temp = df[df["train-size"] == train_size]

    # mean values
    pivot_mean = temp.pivot(
        index="splits",
        columns="C",
        values=metric_mean
    ) * 100

    # std values
    pivot_std = temp.pivot(
        index="splits",
        columns="C",
        values=metric_std
    ) * 100

    # annotations
    annot = pivot_mean.copy().astype(str)

    for i in range(pivot_mean.shape[0]):
        for j in range(pivot_mean.shape[1]):

            mean_val = pivot_mean.iloc[i, j]
            std_val = pivot_std.iloc[i, j]

            annot.iloc[i, j] = (
                f"{mean_val:.1f}\n±{std_val:.1f}"
            )

    sns.heatmap(
        pivot_mean,
        ax=ax,
        annot=annot,
        fmt="",
        cmap="viridis",

        vmin=vmin,
        vmax=vmax,

        square=True,

        linewidths=0.5,
        linecolor='gray',

        cbar=False
    )

    ax.set_title(f"Train size = {train_size}")

    ax.set_xlabel("C")
    ax.set_ylabel("Splits")

# -------------------------------------------------
# 6. Global title
# -------------------------------------------------
fig.suptitle(
    f"{model_name} — Accuracy (%)",
    fontsize=18
)

# -------------------------------------------------
# 7. Save
# -------------------------------------------------
plt.savefig(
    f"D:\Muws\svm\charts\{fileID}_heatmap_{model_name}.png",
    dpi=300,
    bbox_inches="tight"
)


print(f"Figure heatmap saved")

# -------------------------------------------------
# 8. Line plot for fixed C and different train sizes
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
selected_C = 10.0

# только нужный C
temp_df = df_line[
    df_line["C"] == selected_C
].copy()

# тип регуляризации
temp_df["penalty"] = temp_df["model"].apply(
    lambda x: "l1" if "l1" in x.lower() else "l2"
)

# figure
plt.figure(figsize=(10, 6))

# все train sizes
all_train_sizes = sorted(
    temp_df["train-size"].unique()
)

# линии
for penalty in ["l1", "l2"]:

    for train_size in all_train_sizes:

        temp = temp_df[
            (temp_df["penalty"] == penalty) &
            (temp_df["train-size"] == train_size)
        ].sort_values("splits")

        # стили для l1/l2
        if penalty == "l1":
            linestyle = "--"
            marker = "s"
        else:
            linestyle = "-"
            marker = "o"

        plt.plot(
            temp["splits"],

            temp[metric_mean] * 100,

            #yerr=temp[metric_std] * 100,

            color=train_colors[train_size],

            linestyle=linestyle,
            marker=marker,

            linewidth=1,
            markersize=3,

            #capsize=2,

            label=f"{penalty} train={train_size}"
        )

# оформление
plt.title(
    f"Accuracy vs Splits (C={selected_C})"
)

plt.xlabel("Splits")
plt.ylabel("Accuracy (%)")

plt.grid(True)

plt.legend(
    fontsize=8,
    ncol=2
)

# сохранение
plt.savefig(
    f"D:\\Muws\\svm\\charts\\{fileID}_line_C{selected_C}_splits.png",
    dpi=300,
    bbox_inches="tight"
)

print("Line plot saved")

plt.show()
