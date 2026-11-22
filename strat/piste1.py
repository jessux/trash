import pandas as pd, numpy as np
from portfolio import load
def rm(df,P=72,K=2,S=3,fee=0.0005,lo=0.8,hi=1.2,ema=None,hours=None):
    m=df.c.rolling(P).mean();s=df.c.rolling(P).std();up=(m+K*s).values;dn=(m-K*s).values;mid=m.values
    E=df.c.ewm(span=ema).mean().values if ema else None
    hr=df.index.hour.values
    a=df.atr.values;c=df.c.values;h=df.h.values;l=df.l.values;rl=df.rel.values
    out=np.zeros(len(df));pos=None
    for i in range(len(df)):
        if pos:
            d,en,sl,rk,t0=pos;ex=None
            if d==1: ex=sl if l[i]<=sl else mid[i] if h[i]>=mid[i] else None
            else: ex=sl if h[i]>=sl else mid[i] if l[i]<=mid[i] else None
            if ex is None and i-t0>=48: ex=c[i]
            if ex is not None: out[i]=(d*(ex-en)-2*fee*en)/rk;pos=None
        if pos or np.isnan(up[i]) or np.isnan(a[i]) or np.isnan(rl[i]) or not(lo<=rl[i]<=hi): continue
        if hours and not(hours[0]<=hr[i]<hours[1]): continue
        d=1 if c[i]<dn[i] else -1 if c[i]>up[i] else 0
        if ema and d and d*(c[i]-E[i])<0 and False: pass
        if ema and d==1 and c[i]<E[i]: d=0          # achat seulement en tendance haussière
        if ema and d==-1 and c[i]>E[i]: d=0         # vente seulement en tendance baissière
        if d: rk=S*a[i];pos=(d,c[i],c[i]-d*rk,rk,i)
    return pd.Series(out,index=df.index)
df=load("BTCUSDT");mid=df.index[len(df)//2]
V={"base":{},"tendance EMA200h":dict(ema=200),"tendance EMA600h":dict(ema=600),
   "heures 7-20 UTC":dict(hours=(7,20)),"heures 20-24 UTC":dict(hours=(20,24)),
   "EMA200h + 7-20":dict(ema=200,hours=(7,20))}
print(f"{'variante':18s} | TRAIN n  R/trade totalR | TEST n  R/trade totalR")
for k,v in V.items():
    r=rm(df,**v);a=r[r.index<mid];b=r[r.index>=mid];a=a[a!=0];b=b[b!=0]
    print(f"{k:18s} | {len(a):4d} {a.mean():+.3f} {a.sum():+6.1f} | {len(b):4d} {b.mean():+.3f} {b.sum():+6.1f}")
