import pandas as pd, numpy as np
RISK=0.005; RR=2.0; DAILY_STOP=0.03; N=48; ATRM=1.5; FEE=0.0005  # fee+slippage par côté
df=pd.read_csv("BTCUSDT_15m.csv",index_col=0,parse_dates=True)
tr=pd.concat([df.h-df.l,(df.h-df.c.shift()).abs(),(df.l-df.c.shift()).abs()],axis=1).max(axis=1)
df["atr"]=tr.rolling(14).mean()
df["hi"]=df.h.rolling(N).max().shift(); df["lo"]=df.l.rolling(N).min().shift()
df["ema"]=df.c.ewm(span=800).mean()  # ~EMA200 en 1h
def run(df,start_eq=1.0):
    eq=start_eq; pos=None; day=None; day_eq=eq; halted=False; trades=[]; curve=[]
    for t,r in df.iterrows():
        if t.date()!=day: day=t.date(); day_eq=eq; halted=False
        if pos:
            ex=None
            if pos["d"]==1:
                if r.l<=pos["sl"]: ex=pos["sl"]
                elif r.h>=pos["tp"]: ex=pos["tp"]
            else:
                if r.h>=pos["sl"]: ex=pos["sl"]
                elif r.l<=pos["tp"]: ex=pos["tp"]
            if ex:
                pnl=pos["d"]*(ex-pos["e"])/pos["risk"]*RISK*pos["eq"] - 2*FEE*pos["e"]/pos["risk"]*RISK*pos["eq"]
                eq+=pnl; trades.append(pnl/pos["eq"]); pos=None
        if eq<day_eq*(1-DAILY_STOP): halted=True
        if not pos and not halted and not np.isnan(r.atr) and not np.isnan(r.hi):
            d=1 if (r.c>r.hi and r.c>r.ema) else -1 if (r.c<r.lo and r.c<r.ema) else 0
            if d:
                risk=ATRM*r.atr; pos=dict(d=d,e=r.c,sl=r.c-d*risk,tp=r.c+d*RR*risk,risk=risk,eq=eq)
        curve.append(eq)
    return pd.Series(curve,index=df.index),np.array(trades)
c,tr_=run(df)
dd=(c/c.cummax()-1).min()
print(f"trades {len(tr_)} winrate {(tr_>0).mean():.1%} return {c.iloc[-1]-1:.1%} maxDD {dd:.1%}")
m=c.resample("30D").last().pct_change().dropna(); print("30j: moy %.1f%% pire %.1f%%"%(m.mean()*100,m.min()*100))
