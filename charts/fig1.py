import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ===================== SETTINGS =====================
size = "30k"  # "20k", "30k" or "40k"
f = r"D:\Projects\explainable-highdim-svm-recognition\charts\res002-3_aggregated.xlsx"
out = rf"D:\Projects\explainable-highdim-svm-recognition\charts\accuracy_vs_time_{size}.svg"
n_obj = int(size[:-1]) * 1000
# ====================================================

hp, sv, bg = [pd.read_excel(f, s) for s in [f"HP-DSC-{size}", "SVM", f"Bagging-{size}"]]

l1 = lambda d: d.model.astype(str).str.contains("penalty='l1'", regex=False)

hp = hp[hp.n_proc.eq(1)]
bg1, bg2 = bg[bg.n_proc.eq(1) & l1(bg)], bg[bg.n_proc.eq(1) & ~l1(bg)]
s = sv[sv.nTr_obj.eq(n_obj)]
s1, s2 = s[l1(s)].iloc[0], s[~l1(s)].iloc[0]

colors = {"HP-DSC": "#1f77b4", "Bagging": "#ff7f0e", "SVM": "#2ca02c"}
symbols = {"HP-DSC": "circle", "Bagging": "square", "SVM": "diamond"}

fig = make_subplots(rows=1, cols=2, subplot_titles=["L1", "L2"], horizontal_spacing=.08)

def add(d, x, y, xe, ye, name, col):
    fig.add_trace(go.Scatter(x=d[x], y=d[y], mode="markers", name=name,
        marker=dict(size=8, color=colors[name], symbol=symbols[name]),
        error_x=dict(type="data", array=d[xe], visible=True, color=colors[name]),
        error_y=dict(type="data", array=d[ye], visible=True, color=colors[name]),
        hoverinfo="skip"), row=1, col=col)

add(hp[l1(hp)], "tr_time_mean", "acc_tst_mean", "tr_time_std", "acc_tst_std", "HP-DSC", 1)
add(bg1, "tr_time_mean", "acc_mean", "tr_time_std", "acc_std", "Bagging", 1)
add(bg2, "tr_time_mean", "acc_mean", "tr_time_std", "acc_std", "Bagging", 2)

for col, s in [(1, s1), (2, s2)]:
    fig.add_hline(y=s.acc_mean, row=1, col=col,
                  line=dict(color=colors["SVM"], dash="dash", width=1.5))
    if col == 2:
        fig.add_trace(go.Scatter(x=[s.tr_time_mean], y=[s.acc_mean], mode="markers",
            name="SVM", marker=dict(size=8, color=colors["SVM"], symbol=symbols["SVM"]),
            error_x=dict(type="data", array=[s.tr_time_std], visible=True, color=colors["SVM"]),
            error_y=dict(type="data", array=[s.acc_std], visible=True, color=colors["SVM"]),
            hoverinfo="skip"), row=1, col=col)
    fig.add_annotation(x=.98, y=s.acc_mean,
        xref=f"x{'' if col == 1 else 2} domain",
        text=f"{s.acc_mean:.4f}  |  {s.tr_time_mean:.1f} s",
        showarrow=False, xanchor="right", yanchor="bottom",
        font=dict(color=colors["SVM"]), row=1, col=col)

fig.update_xaxes(title_text="Training time, s", showgrid=True, zeroline=False)
fig.update_yaxes(title_text="Accuracy", showgrid=True, zeroline=False)
fig.update_layout(width=1100, height=520, template="plotly_white",
                  font=dict(size=13), legend=dict(orientation="h", y=1.08, x=0),
                  margin=dict(l=70, r=30, t=70, b=60))

fig.write_image(out, width=1100, height=520, scale=2)
fig.show()