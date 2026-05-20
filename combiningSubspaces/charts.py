import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------
# 1. Load aggregated sheet
# -------------------------------------------------
file = r"D:\Cloud\SVM\s1_synthetic1-10k-f1000-i1000-r0-l0.npz.xlsx"

df = pd.read_excel(
    file,
    sheet_name="aggregated"
)

# -------------------------------------------------
# 2. Parameters
# -------------------------------------------------
metric_mean = "Acc(test)_mean"
metric_std = "Acc(test)_std"

model_name = "Comb-LSVC-l2"

train_sizes = [250, 500, 1000]

save_file = f"D:\Cloud\SVM\charts\heatmap_{model_name}.png"

# -------------------------------------------------
# 3. Filter dataframe
# -------------------------------------------------
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
        index="C",
        columns="splits",
        values=metric_mean
    ) * 100

    # std values
    pivot_std = temp.pivot(
        index="C",
        columns="splits",
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

    ax.set_xlabel("Splits")
    ax.set_ylabel("C")

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
    save_file,
    dpi=300,
    bbox_inches="tight"
)

print(f"Figure saved: {save_file}")

plt.show()