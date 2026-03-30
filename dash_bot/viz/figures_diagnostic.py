# dash_bot/viz/figures_diagnostic.py
"""
Cycle-level diagnostic visualizations for the copula pairs trading dashboard.
"""
from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from .figures import TRON_LAYOUT, fig_empty

CYAN="#00f0ff"; MAGENTA="#ff2eed"; GREEN="#00ff88"; RED="#ff3355"
ORANGE="#ff6f1a"; GOLD="#ffe01a"; PURPLE="#7b61ff"; SUBTEXT="#5a7a90"
GRID_COLOR="rgba(0,240,255,0.06)"; ZONE_CLOSE="rgba(255,228,26,0.10)"

def _tron_layout(**overrides):
    out=dict(TRON_LAYOUT); out.update(overrides); return out
def _subplot_base():
    return {k:v for k,v in TRON_LAYOUT.items() if k not in ("xaxis","yaxis","legend","margin")}

# ═══════════════════════════════════════════════════════════════════
# 1. COPULA DECISION MAP
# ═══════════════════════════════════════════════════════════════════
def fig_copula_decision_map(cop,u_trades,v_trades,signals,entry,exit_thr,copula_name="",pair_label="",Ngrid=80):
    fig=go.Figure(); eps=1e-4; grid=np.linspace(eps,1-eps,Ngrid); U,V=np.meshgrid(grid,grid)
    H12=np.full_like(U,0.5); H21=np.full_like(U,0.5)
    for i in range(Ngrid):
        for j in range(Ngrid):
            try:
                if hasattr(cop,"conditional_cdf_u_given_v"):
                    H12[i,j]=float(cop.conditional_cdf_u_given_v(float(U[i,j]),float(V[i,j])))
                    H21[i,j]=float(cop.conditional_cdf_v_given_u(float(U[i,j]),float(V[i,j])))
            except: pass
    Z=np.zeros_like(U)
    Z[(H12>1-entry)&(H21<entry)]=1; Z[(H12<entry)&(H21>1-entry)]=-1
    Z[(np.abs(H12-0.5)<exit_thr)&(np.abs(H21-0.5)<exit_thr)]=2
    fig.add_trace(go.Heatmap(x=grid,y=grid,z=Z,colorscale=[[0,RED],[0.33,"rgba(10,22,40,0.95)"],[0.66,GREEN],[1,GOLD]],zmin=-1,zmax=2,showscale=False,opacity=0.35,hoverinfo="skip"))
    try:
        Zp=np.array([[float(cop.get_pdf(float(U[i,j]),float(V[i,j]))) for j in range(Ngrid)] for i in range(Ngrid)])
        Zp=np.clip(Zp,0,np.percentile(Zp[Zp>0],98) if (Zp>0).any() else 1)
        fig.add_trace(go.Contour(x=grid,y=grid,z=Zp,ncontours=12,colorscale="Blues",showscale=False,opacity=0.20,line=dict(width=0.5,color="rgba(0,240,255,0.15)"),hoverinfo="skip"))
    except: pass
    for za,st,pf in [(H12,"dot","h12"),(H21,"dash","h21")]:
        fig.add_trace(go.Contour(x=grid,y=grid,z=za,contours=dict(start=entry,end=entry,size=0.01,coloring="none"),line=dict(color=RED,width=1.5,dash=st),showscale=False,hoverinfo="skip",name=f"{pf}={entry:.2f}"))
        fig.add_trace(go.Contour(x=grid,y=grid,z=za,contours=dict(start=1-entry,end=1-entry,size=0.01,coloring="none"),line=dict(color=GREEN,width=1.5,dash=st),showscale=False,hoverinfo="skip",name=f"{pf}={1-entry:.2f}"))
    for val in [0.5-exit_thr,0.5+exit_thr]:
        for za in [H12,H21]:
            fig.add_trace(go.Contour(x=grid,y=grid,z=za,contours=dict(start=val,end=val,size=0.01,coloring="none"),line=dict(color=GOLD,width=1,dash="dot"),showscale=False,hoverinfo="skip",showlegend=False))
    if len(u_trades)>0:
        sig=np.asarray(signals)
        fig.add_trace(go.Scatter(x=u_trades,y=v_trades,mode="lines",line=dict(color="rgba(200,220,232,0.15)",width=0.8),hoverinfo="skip",showlegend=False))
        for mask,col,nm,sym in [(sig==0,SUBTEXT,"Hold","circle"),(sig==1,GREEN,"Long C1","triangle-up"),(sig==-1,RED,"Short C1","triangle-down")]:
            if mask.any():
                fig.add_trace(go.Scatter(x=u_trades[mask],y=v_trades[mask],mode="markers",name=nm,marker=dict(size=4 if nm=="Hold" else 10,color=col,symbol=sym,opacity=0.4 if nm=="Hold" else 1,line=dict(width=1,color="white") if nm!="Hold" else dict(width=0)),hovertemplate=f"u=%{{x:.3f}} v=%{{y:.3f}}<extra>{nm}</extra>"))
    title=f"⬡  DECISION MAP — {copula_name}"
    if pair_label: title+=f"  [{pair_label}]"
    fig.update_layout(**_tron_layout(height=520,width=560,title=title,xaxis=dict(title="u₁",range=[0,1],gridcolor=GRID_COLOR,tickfont=dict(color=SUBTEXT)),yaxis=dict(title="u₂",range=[0,1],gridcolor=GRID_COLOR,tickfont=dict(color=SUBTEXT),scaleanchor="x",scaleratio=1),legend=dict(x=0.01,y=0.99,bgcolor="rgba(10,22,40,0.8)",font=dict(size=9))))
    return fig

# ═══════════════════════════════════════════════════════════════════
# 2. H-FUNCTIONS TIME SERIES
# ═══════════════════════════════════════════════════════════════════
def fig_h_functions_timeseries(timestamps,h12_series,h21_series,signals,entry,exit_thr,pair_label=""):
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,subplot_titles=["h₁|₂ = P(U₁≤u₁ | U₂=u₂)","h₂|₁ = P(U₂≤u₂ | U₁=u₁)"],vertical_spacing=0.08)
    for row,(h,nm,c) in enumerate([(h12_series,"h₁|₂",CYAN),(h21_series,"h₂|₁",MAGENTA)],1):
        fig.add_trace(go.Scatter(x=timestamps,y=h,name=nm,mode="lines",line=dict(color=c,width=1.8)),row=row,col=1)
        fig.add_hline(y=entry,line=dict(color=RED,width=1,dash="dash"),annotation_text=f"α₁={entry}",row=row,col=1)
        fig.add_hline(y=1-entry,line=dict(color=GREEN,width=1,dash="dash"),annotation_text=f"1-α₁={1-entry:.2f}",row=row,col=1)
        fig.add_hrect(y0=0.5-exit_thr,y1=0.5+exit_thr,fillcolor=ZONE_CLOSE,line_width=0,annotation_text="close zone",row=row,col=1)
        fig.add_hline(y=0.5,line=dict(color=SUBTEXT,width=0.5,dash="dot"),row=row,col=1)
    sig=np.asarray(signals); ts=np.asarray(timestamps)
    for mask,c,nm,sym in [(sig==1,GREEN,"Long C1","triangle-up"),(sig==-1,RED,"Short C1","triangle-down")]:
        if mask.any(): fig.add_trace(go.Scatter(x=ts[mask],y=h12_series[mask],mode="markers",name=nm,marker=dict(size=9,color=c,symbol=sym,line=dict(width=1,color="white"))),row=1,col=1)
    title=f"⬡  H-FUNCTIONS — {pair_label}" if pair_label else "⬡  H-FUNCTIONS"
    fig.update_layout(**_subplot_base(),height=480,title=title,margin=dict(l=35,r=25,t=55,b=35),legend=dict(x=1.02,y=1,font=dict(size=9)))
    fig.update_yaxes(range=[-0.02,1.02])
    return fig

# ═══════════════════════════════════════════════════════════════════
# 3. SPREAD + TRADES
# ═══════════════════════════════════════════════════════════════════
def fig_spread_with_trades(spread1_form,spread2_form,spread1_trade,spread2_trade,trade_entries,trade_exits,pair_label="",beta1=np.nan,beta2=np.nan):
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,subplot_titles=["Spread₁ = BTC − β₁·P₁","Spread₂ = BTC − β₂·P₂"],vertical_spacing=0.08)
    for row,(sf,st,c,lb) in enumerate([(spread1_form,spread1_trade,CYAN,"S₁"),(spread2_form,spread2_trade,MAGENTA,"S₂")],1):
        if sf is not None and len(sf)>0: fig.add_trace(go.Scatter(x=sf.index,y=sf.values,name=f"{lb} (form)",mode="lines",line=dict(color=c,width=1.2,dash="dot"),opacity=0.5),row=row,col=1)
        if st is not None and len(st)>0: fig.add_trace(go.Scatter(x=st.index,y=st.values,name=f"{lb} (trade)",mode="lines",line=dict(color=c,width=2)),row=row,col=1)
    for ed in trade_entries:
        t=pd.to_datetime(ed.get("time")); d=ed.get("direction",""); c=GREEN if "LONG1" in d else RED
        fig.add_vline(x=t,line=dict(color=c,width=1.5),row="all",col=1)
    for ex in trade_exits: fig.add_vline(x=pd.to_datetime(ex.get("time")),line=dict(color=GOLD,width=1,dash="dot"),row="all",col=1)
    title=f"⬡  SPREADS — {pair_label}" if pair_label else "⬡  SPREADS"
    if np.isfinite(beta1) and np.isfinite(beta2): title+=f"  (β₁={beta1:.2f}, β₂={beta2:.2f})"
    fig.update_layout(**_subplot_base(),height=420,title=title,margin=dict(l=35,r=25,t=55,b=35))
    return fig

# ═══════════════════════════════════════════════════════════════════
# 4. CYCLE SUMMARY CARD
# ═══════════════════════════════════════════════════════════════════
def fig_cycle_summary_card(cycle_id,pair,copula_name,beta1,beta2,q1,q2,n_trades,cycle_pnl,entry_threshold,exit_threshold,status="OK"):
    fig=go.Figure()
    lines=[f"<b>Cycle {cycle_id}</b>     Status: <b>{status}</b>",f"Pair: <b>{pair}</b>     Copula: <b>{copula_name}</b>",f"β₁ = {beta1:.4f}     β₂ = {beta2:.4f}",f"Q₁ = {q1:.2f}     Q₂ = {q2:.2f}",f"Trades: <b>{n_trades}</b>     PnL: <b style='color:{GREEN if cycle_pnl>=0 else RED}'>{cycle_pnl:+,.0f} USDT</b>",f"Entry α₁ = {entry_threshold}     Exit α₂ = {exit_threshold}"]
    fig.add_annotation(x=0.5,y=0.5,xref="paper",yref="paper",text="<br>".join(lines),showarrow=False,font=dict(family="Share Tech Mono",size=13,color="#c8dce8"),align="left",bordercolor=CYAN,borderwidth=1,borderpad=16,bgcolor="rgba(10,22,40,0.95)")
    fig.update_layout(**_tron_layout(height=200,xaxis=dict(visible=False),yaxis=dict(visible=False),margin=dict(l=10,r=10,t=10,b=10)))
    return fig

# ═══════════════════════════════════════════════════════════════════
# 5. PSEUDO-OBS SCATTER WITH COPULA PDF OVERLAY
# ═══════════════════════════════════════════════════════════════════
def fig_pseudo_obs_scatter(pseudo_u,pseudo_v,overlay_cop=None,overlay_name="",selected_name="",pair_label="",Ngrid=60):
    """Scatter of (u,v) pseudo-observations with optional copula PDF contour overlay."""
    fig=go.Figure()
    if overlay_cop is not None:
        eps=1e-3; grid=np.linspace(eps,1-eps,Ngrid)
        try:
            Z=np.array([[float(overlay_cop.get_pdf(float(u),float(v))) for u in grid] for v in grid])
            Z=np.clip(Z,0,np.percentile(Z[Z>0],97) if (Z>0).any() else 1)
            fig.add_trace(go.Contour(x=grid,y=grid,z=np.log1p(Z),ncontours=15,colorscale=[[0,"rgba(10,22,40,0)"],[0.3,"rgba(0,100,200,0.15)"],[0.6,"rgba(0,180,255,0.3)"],[1.0,"rgba(0,240,255,0.5)"]],showscale=False,line=dict(width=0.8,color="rgba(0,240,255,0.25)"),hoverinfo="skip",name=f"PDF: {overlay_name}"))
        except: pass
    if len(pseudo_u)>0:
        fig.add_trace(go.Scatter(x=np.asarray(pseudo_u),y=np.asarray(pseudo_v),mode="markers",name="Pseudo-obs (formation)",marker=dict(size=3.5,color=MAGENTA,opacity=0.5,line=dict(width=0)),hovertemplate="u=%{x:.3f}<br>v=%{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",showlegend=False,line=dict(color=SUBTEXT,width=0.5,dash="dot"),hoverinfo="skip"))
    title="⬡  PSEUDO-OBSERVATIONS"
    if overlay_name:
        badge=" ★ SELECTED" if overlay_name==selected_name else ""
        title+=f" — {overlay_name}{badge}"
    if pair_label: title+=f"  [{pair_label}]"
    fig.update_layout(**_tron_layout(height=500,width=520,title=title,xaxis=dict(title="u₁ = ECDF(S₁)",range=[0,1],gridcolor=GRID_COLOR,tickfont=dict(color=SUBTEXT)),yaxis=dict(title="u₂ = ECDF(S₂)",range=[0,1],gridcolor=GRID_COLOR,tickfont=dict(color=SUBTEXT),scaleanchor="x",scaleratio=1),legend=dict(x=0.01,y=0.99,bgcolor="rgba(10,22,40,0.8)",font=dict(size=9))))
    return fig

# ═══════════════════════════════════════════════════════════════════
# 6. FIT COMPARISON BAR CHART
# ═══════════════════════════════════════════════════════════════════
def fig_fit_ranking(fit_summary,selected_name="",metric="aic",top_n=20):
    """Horizontal bar chart ranking copula candidates by AIC or score."""
    if not fit_summary: return fig_empty("No fit data available")
    df=pd.DataFrame(fit_summary)
    if metric not in df.columns or df[metric].isna().all():
        for m in ["aic","score","loglik"]:
            if m in df.columns and not df[m].isna().all(): metric=m; break
    df=df.dropna(subset=[metric]).copy()
    if df.empty: return fig_empty("No valid fit metrics")
    ascending=metric!="loglik"
    df=df.sort_values(metric,ascending=ascending).head(top_n).iloc[::-1]
    colors=[CYAN if str(r["name"])==str(selected_name) else ("rgba(0,240,255,0.4)" if r.get("evaluable",True) else "rgba(100,100,100,0.3)") for _,r in df.iterrows()]
    fig=go.Figure()
    fig.add_trace(go.Bar(y=df["name"].astype(str),x=df[metric].values,orientation="h",marker=dict(color=colors,line=dict(color=colors,width=1)),hovertemplate="%{y}<br>"+metric.upper()+"=%{x:.2f}<extra></extra>"))
    sel=df[df["name"]==selected_name]
    if not sel.empty: fig.add_annotation(x=float(sel[metric].iloc[0]),y=str(sel["name"].iloc[0]),text=" ★ SELECTED",showarrow=False,font=dict(color=GOLD,size=10,family="Share Tech Mono"),xanchor="left")
    label={"aic":"AIC (lower = better)","score":"Score (lower = better)","loglik":"Log-Lik (higher = better)"}.get(metric,metric.upper())
    fig.update_layout(**_tron_layout(height=max(300,len(df)*22+80),title=f"⬡  COPULA FIT RANKING — {label}",xaxis=dict(title=label,gridcolor=GRID_COLOR,tickfont=dict(color=SUBTEXT)),yaxis=dict(tickfont=dict(color=SUBTEXT,size=9)),margin=dict(l=140,r=25,t=50,b=35)))
    return fig

# ═══════════════════════════════════════════════════════════════════
# 7. TAIL DEPENDENCE COMPARISON
# ═══════════════════════════════════════════════════════════════════
def fig_tail_dependence(fit_summary,selected_name="",top_n=12):
    """Grouped bar: lower/upper tail dependence for top candidates vs empirical."""
    if not fit_summary: return fig_empty("No tail data")
    df=pd.DataFrame(fit_summary)
    if not {"tail_dep_L","tail_dep_U"}.issubset(df.columns): return fig_empty("No tail dependence data")
    df=df.dropna(subset=["tail_dep_L","tail_dep_U"]).head(top_n)
    if df.empty: return fig_empty("No tail dependence data")
    fig=go.Figure()
    fig.add_trace(go.Bar(name="λ_L (lower)",x=df["name"].astype(str),y=df["tail_dep_L"].values,marker=dict(color="rgba(0,180,255,0.7)",line=dict(color=CYAN,width=1))))
    fig.add_trace(go.Bar(name="λ_U (upper)",x=df["name"].astype(str),y=df["tail_dep_U"].values,marker=dict(color="rgba(255,46,237,0.7)",line=dict(color=MAGENTA,width=1))))
    fig.update_layout(**_tron_layout(height=350,barmode="group",title="⬡  TAIL DEPENDENCE — Model vs Empirical",xaxis=dict(tickfont=dict(color=SUBTEXT,size=8),tickangle=-45),yaxis=dict(title="Tail dep λ",gridcolor=GRID_COLOR,tickfont=dict(color=SUBTEXT)),legend=dict(x=0.7,y=0.95,font=dict(size=9)),margin=dict(l=50,r=25,t=50,b=80)))
    return fig