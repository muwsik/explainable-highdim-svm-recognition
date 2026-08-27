import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

size = "20k"  # "20k", "30k", "40k"
f = r"D:\Projects\explainable-highdim-svm-recognition\charts\res002-3_aggregated.xlsx"
out = rf"D:\Projects\explainable-highdim-svm-recognition\charts\accuracy_vs_processes_{size}.svg"
procs = [1,2,4,8]

hp,bg = [pd.read_excel(f,s) for s in [f"HP-DSC-{size}",f"Bagging-{size}"]]
l1 = lambda d: d.model.astype(str).str.contains("penalty='l1'",regex=False)

hp = hp[hp.n_proc.isin(procs) & hp.n_splits.eq(500)]
bg = bg[bg.n_proc.isin(procs) & bg.max_feats.eq(500) & bg.n_est.eq(200)]

hp_l1,hp_l2 = hp[l1(hp)],hp[~l1(hp)]
bg_l1,bg_l2 = bg[l1(bg)],bg[~l1(bg)]

colors = {"HP-DSC":"#1f77b4","Bagging":"#ff7f0e"}

fig = make_subplots(rows=1,cols=2,subplot_titles=["L1","L2"],horizontal_spacing=.08)

def add(d,name,col,y="acc_mean",std="acc_std",dash="solid"):
    if d.empty: return
    d=d.sort_values("n_proc")
    fig.add_trace(go.Scatter(x=d.n_proc,y=d[y],mode="lines+markers",name=name,
        line=dict(color=colors[name],width=2,dash=dash),
        marker=dict(size=8,color=colors[name],
                    symbol="circle" if name=="HP-DSC" else "square"),
        error_y=dict(type="data",array=d[std],visible=True,color=colors[name]),
        hoverinfo="skip"),row=1,col=col)

add(hp_l1,"HP-DSC",1,"acc_tst_mean","acc_tst_std","dash")
add(bg_l1,"Bagging",1)
add(hp_l2,"HP-DSC",2,"acc_tst_mean","acc_tst_std","dash")
add(bg_l2,"Bagging",2)

fig.update_xaxes(title_text="Number of processes",tickmode="array",tickvals=procs,
                 showgrid=True,zeroline=False)
fig.update_yaxes(title_text="Accuracy",showgrid=True,zeroline=False)
fig.update_layout(width=1100,height=520,template="plotly_white",font=dict(size=13),
                  legend=dict(orientation="h",y=1.08,x=0),
                  margin=dict(l=70,r=30,t=70,b=60))

fig.write_image(out,width=1100,height=520,scale=2)
fig.show()