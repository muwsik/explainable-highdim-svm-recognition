import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sizes=["20k", "30k", "40k"]
dsc_splits=[10, 100, 500]
bagging_params=[(30,50), (50,250), (200,500)]

f=r"C:\Users\Victory\.vscode\repos\explainable-highdim-svm-recognition\charts\res002-4_aggregated.xlsx"
out=r"C:\Users\Victory\.vscode\repos\explainable-highdim-svm-recognition\charts\time_vs_dataset.svg"

colors={"HP-DSC":"#1f77b4","Bagging":"#ff7f0e"}
dashes=["dot","dash","solid"]

hp={s:pd.read_excel(f,f"HP-DSC-{s}") for s in sizes}
bg={s:pd.read_excel(f,f"Bagging-{s}") for s in sizes}

def is_l1(d):
    return d.model.astype(str).str.contains("penalty='l1'",regex=False)

fig=make_subplots(rows=1,cols=2,subplot_titles=["L1","L2"],horizontal_spacing=.08)

def add(x,y,e,name,label,col,dash,symbol):
    if not all(pd.notna(v) for v in y): return
    fig.add_trace(go.Scatter(
        x=x,y=y,error_y=dict(type="data",array=e,visible=True),
        mode="lines+markers",name=label,showlegend=col==1,
        line=dict(color=colors[name],width=0.8,dash=dash),
        marker=dict(size=5,color=colors[name],symbol=symbol),
        hoverinfo="skip"),row=1,col=col)

for col,l1 in [(1,True),(2,False)]:

    for j,ns in enumerate(dsc_splits):
        vals=[]; errs=[]
        for s in sizes:
            d=hp[s]
            d=d[(d.n_proc==1)&(d.n_splits==ns)&
                (is_l1(d) if l1 else ~is_l1(d))]
            vals.append(d.tr_time_mean.iloc[0] if not d.empty else None)
            errs.append(d.tr_time_std.iloc[0] if not d.empty else None)

        add(sizes,vals,errs,"HP-DSC",
            f"HP-DSC (n_splits={ns})",col,dashes[j],"circle")

    for j,(ne,mf) in enumerate(bagging_params):
        vals=[]; errs=[]
        for s in sizes:
            d=bg[s]
            d=d[(d.n_proc==1)&(d.n_est==ne)&(d.max_feats==mf)&
                (is_l1(d) if l1 else ~is_l1(d))]
            vals.append(d.tr_time_mean.iloc[0] if not d.empty else None)
            errs.append(d.tr_time_std.iloc[0] if not d.empty else None)

        add(sizes,vals,errs,"Bagging",
            f"Bagging (n_est={ne}, max_feats={mf})",
            col,dashes[j],"square")

fig.update_xaxes(title_text="Dataset size",showgrid=True,zeroline=False)
fig.update_yaxes(title_text="Training time, s",showgrid=True,zeroline=False)

fig.update_layout(
    width=1150,height=540,template="plotly_white",font=dict(size=13),
    legend=dict(orientation="h",y=1.08,x=0),
    margin=dict(l=70,r=30,t=70,b=60))

fig.write_image(out,width=1150,height=540,scale=2)