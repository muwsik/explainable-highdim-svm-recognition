import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

size="20k"                  # "20k","30k","40k"
metric="time"               # "acc" или "time"
est_values=None        # None = все n_est
dsc_splits=[50,100,200,500] # None = все n_splits

f=r"C:\Users\Victory\.vscode\repos\explainable-highdim-svm-recognition\charts\res002-4_aggregated.xlsx"
out=rf"C:\Users\Victory\.vscode\repos\explainable-highdim-svm-recognition\charts\{metric}_vs_subspace_{size}.svg"

hp,bg,sv=[pd.read_excel(f,s) for s in [f"HP-DSC-{size}",f"Bagging-{size}","SVM"]]
l1=lambda d:d.model.astype(str).str.contains("penalty='l1'",regex=False)

n_feat=int(size[:-1])*1000
hp["subspace"]=n_feat/hp.n_splits
hp=hp[hp.n_proc.eq(1)]
bg=bg[bg.n_proc.eq(1)]
sv=sv[sv.nTr_obj.eq(n_feat)]

if dsc_splits is not None: hp=hp[hp.n_splits.isin(dsc_splits)]
if est_values is not None: bg=bg[bg.n_est.isin(est_values)]

cols={"acc":("acc_mean","acc_std","Accuracy"),
      "time":("tr_time_mean","tr_time_std","Training time, s")}
bg_y,bg_std,ylabel=cols[metric]
hp_y="acc_tst_mean" if metric=="acc" else "tr_time_mean"
hp_std="acc_tst_std" if metric=="acc" else "tr_time_std"

colors={"HP-DSC":"#1f77b4","Bagging":"#ff7f0e","SVM":"#2ca02c"}
dashes={10:"dot",50:"dash",100:"dashdot",200:"solid"}

fig=make_subplots(rows=1,cols=2,subplot_titles=["L1","L2"],horizontal_spacing=.08)

def add(d,name,col,xcol,ycol,stdcol,symbol="circle",showlegend=False,dash="solid"):
    if d.empty:return
    d=d.sort_values(xcol); c=colors[name]
    fig.add_trace(go.Scatter(x=d[xcol],y=d[ycol],mode="lines+markers",
        name=name,showlegend=showlegend,line=dict(color=c,width=0.8,dash=dash),
        marker=dict(size=5,color=c,symbol=symbol),
        error_y=dict(type="data",array=d[stdcol],visible=True,color=c),
        hoverinfo="skip"),row=1,col=col)

for col,mask,s in [(1,l1,sv[l1(sv)]),(2,lambda d:~l1(d),sv[~l1(sv)])]:
    h=hp[mask(hp)]
    add(h,"HP-DSC",col,"subspace",hp_y,hp_std,"circle",True)

    b=bg[mask(bg)]
    for i,ne in enumerate(sorted(b.n_est.unique())):
        add(b[b.n_est.eq(ne)],"Bagging",col,"max_feats",bg_y,bg_std,
            "square",i==0,dashes.get(ne,"solid"))

    if not s.empty:
        s=s.iloc[0]
        sy=s[bg_y]

        if metric=="time" and col==1:
            vals=pd.concat([h[hp_y],b[bg_y]]).dropna()
            sy=vals.max()*1.05 if not vals.empty else sy

        fig.add_hline(y=sy,row=1,col=col,
            line=dict(color=colors["SVM"],dash="dot",width=1.5))

        fig.add_annotation(x=.98,y=sy,
            xref=f"x{'' if col==1 else 2} domain",
            text=f"SVM: {s[bg_y]:.2f} s",
            showarrow=False,xanchor="right",yanchor="bottom",
            font=dict(color=colors["SVM"]),row=1,col=col)

fig.add_trace(go.Scatter(x=[None],y=[None],mode="lines",name="SVM",
    line=dict(color=colors["SVM"],dash="dot",width=1.5)),row=1,col=1)

fig.update_xaxes(title_text="Features per subspace",showgrid=True,zeroline=False)
fig.update_yaxes(title_text=ylabel,showgrid=True,zeroline=False)
fig.update_layout(width=1150,height=540,template="plotly_white",font=dict(size=13),
    legend=dict(orientation="h",y=1.08,x=0),margin=dict(l=70,r=30,t=70,b=60))

fig.write_image(out,width=1150,height=540,scale=2)