import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

size = "20k"  # "20k", "30k", "40k"
f = r"D:\Projects\explainable-highdim-svm-recognition\charts\res002-3_aggregated.xlsx"
out = rf"D:\Projects\explainable-highdim-svm-recognition\charts\time_vs_processes_{size}.svg"

hp,sv,bg = [pd.read_excel(f,s) for s in [f"HP-DSC-{size}","SVM",f"Bagging-{size}"]]
l1 = lambda d: d.model.astype(str).str.contains("penalty='l1'",regex=False)

hp = hp[hp.n_proc.isin([1,2,4,8])]
hp10,hp500 = hp[hp.n_splits.eq(10)],hp[hp.n_splits.eq(500)]

bg = bg[bg.n_proc.isin([1,2,4,8]) & bg.max_feats.eq(500) & bg.n_est.eq(200)]
bg1,bg2 = bg[l1(bg)],bg[~l1(bg)]

s = sv[sv.nTr_obj.eq(int(size[:-1])*1000)]
s1,s2 = s[l1(s)].iloc[0],s[~l1(s)].iloc[0]

colors = {"HP-DSC":"#1f77b4","Bagging":"#ff7f0e","SVM":"#2ca02c"}

fig = make_subplots(rows=1,cols=2,subplot_titles=["L1","L2"],horizontal_spacing=.08)

def add(d,name,col,dash="solid"):
    fig.add_trace(go.Scatter(x=d.n_proc,y=d.tr_time_mean,mode="lines+markers",name=name,
        line=dict(color=colors[name],width=2,dash=dash),
        marker=dict(size=8,color=colors[name],symbol="circle" if name=="HP-DSC" else "square"),
        error_y=dict(type="data",array=d.tr_time_std,visible=True,color=colors[name]),
        hoverinfo="skip"),row=1,col=col)

for col,mask in [(1,l1),(2,lambda d:~l1(d))]:
    add(hp500[mask(hp500)],"HP-DSC",col,"dash")
    add(bg1 if col==1 else bg2,"Bagging",col)

for col,s in [(1,s1),(2,s2)]:
    if col==1:
        ymax=max(pd.concat([hp10[l1(hp10)].tr_time_mean,hp500[l1(hp500)].tr_time_mean,bg1.tr_time_mean]))
        y=ymax*1.08
    else:
        y=s.tr_time_mean

    fig.add_hline(y=y,row=1,col=col,line=dict(color=colors["SVM"],dash="dot",width=1.5))
    fig.add_annotation(x=.98,y=y,xref=f"x{'' if col==1 else 2} domain",
        text=f"SVM: {s.tr_time_mean:.1f} s",showarrow=False,xanchor="right",
        yanchor="bottom",font=dict(color=colors["SVM"]),row=1,col=col)

    if col==2:
        fig.add_trace(go.Scatter(x=[1],y=[s.tr_time_mean],mode="markers",name="SVM",
            marker=dict(size=9,color=colors["SVM"],symbol="diamond"),
            error_y=dict(type="data",array=[s.tr_time_std],visible=True,color=colors["SVM"]),
            hoverinfo="skip"),row=1,col=col)

fig.update_xaxes(title_text="Number of processes",tickmode="array",tickvals=[1,2,4,8],
                 showgrid=True,zeroline=False)
fig.update_yaxes(title_text="Training time, s",showgrid=True,zeroline=False)
fig.update_layout(width=1100,height=520,template="plotly_white",font=dict(size=13),
                  legend=dict(orientation="h",y=1.08,x=0),margin=dict(l=70,r=30,t=70,b=60))

fig.write_image(out,width=1100,height=520,scale=2)
fig.show()