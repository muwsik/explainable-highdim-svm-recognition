import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

mode="40k"                  # "datasets" или "40k"
splits=50                       # для datasets: 50 или 100
splits_40k=[10,50,100,500]       # варианты для режима 40k
sizes=["20k","30k","40k"]
procs=[1,2,4,6,8,10]

f=r"C:\Users\Victory\.vscode\repos\explainable-highdim-svm-recognition\charts\res002-4_aggregated.xlsx"
out=rf"C:\Users\Victory\.vscode\repos\explainable-highdim-svm-recognition\charts\speedup_dsc_{mode}.svg"

hp={s:pd.read_excel(f,f"HP-DSC-{s}") for s in sizes}

def is_l1(d):
    return d.model.astype(str).str.contains("penalty='l1'",regex=False)

colors={"20k":"#1f77b4","30k":"#ff7f0e","40k":"#2ca02c"}
dashes={10:"dot",50:"dash",100:"dashdot",500:"solid"}

title = (f"HP-DSC speedup (n_splits={splits})"
         if mode=="datasets"
         else "HP-DSC speedup (dataset size = 40k)")

fig=make_subplots(
    rows=1,cols=2,
    subplot_titles=["L1","L2"],
    horizontal_spacing=.08)

fig.update_layout(
    title=dict(text=title,x=.5,xanchor="center"),
    width=1150,height=540,
    template="plotly_white",
    font=dict(size=13),
    legend=dict(orientation="h",y=1.08,x=0),
    margin=dict(l=70,r=30,t=90,b=60))

def add(x,y,name,col,dash="solid",symbol="circle",legend=True):
    if not any(pd.notna(v) for v in y): return
    fig.add_trace(go.Scatter(
        x=x,y=y,mode="lines+markers",name=name,
        showlegend=legend,
        line=dict(color=colors.get(name,"#1f77b4"),width=1.8,dash=dash),
        marker=dict(size=5,symbol=symbol),
        hoverinfo="skip"),row=1,col=col)

if mode=="datasets":
    for col,l1 in [(1,True),(2,False)]:
        for size in sizes:
            d=hp[size]
            d=d[(d.n_splits==splits)&(d.n_proc.isin(procs))&
                (is_l1(d) if l1 else ~is_l1(d))].sort_values("n_proc")
            if d.empty: continue

            base=d.loc[d.n_proc.eq(1),"tr_time_mean"]
            if base.empty: continue

            t1=base.iloc[0]
            speed=[t1/t for t in d.tr_time_mean]
            add(d.n_proc,speed,size,col)

else:
    for col,l1 in [(1,True),(2,False)]:
        for i,ns in enumerate(splits_40k):
            d=hp["40k"]
            d=d[(d.n_splits==ns)&(d.n_proc.isin(procs))&
                (is_l1(d) if l1 else ~is_l1(d))].sort_values("n_proc")
            if d.empty: continue

            base=d.loc[d.n_proc.eq(1),"tr_time_mean"]
            if base.empty: continue

            t1=base.iloc[0]
            speed=[t1/t for t in d.tr_time_mean]
            add(d.n_proc,speed,f"n_splits={ns}",col,
                dashes.get(ns,"solid"),"circle")

fig.update_xaxes(
    title_text="Number of processes",
    tickvals=procs,
    showgrid=True,
    zeroline=False)

fig.update_yaxes(
    title_text="Speedup",
    showgrid=True,
    zeroline=False)

fig.update_layout(
    width=1150,height=540,
    template="plotly_white",
    font=dict(size=13),
    legend=dict(orientation="h",y=1.08,x=0),
    margin=dict(l=70,r=30,t=70,b=60))

fig.write_image(out,width=1150,height=540,scale=2)