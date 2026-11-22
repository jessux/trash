import pandas as pd, numpy as np, itertools
RISK=0.005; DAILY_STOP=0.03
raw=pd.read_csv("BTCUSDT_15m.csv",index_col=0,parse_dates=True)
df=raw.resample("1h").agg(dict(o="first",h="max",l="min",c="last")).dropna()
tr=pd.concat([df.h-df.l,(df.h-df.c.shift()).abs(),(df.l-df.c.shift()).abs()],axis=1).max(axis=1)
df["atr"]=tr.rolling(14).mean(); df["ema"]=df.c.ewm(span=200).mean()
# régime de vol (connu à 00:00 UTC): amplitude de la veille / moyenne 30j
dd=raw.resample("1D").agg(dict(o="first",h="max",l="min"));rng=(dd.h-dd.l)/dd.o
rel=(rng/rng.rolling(30).mean()).shift(1); df["rel"]=rel.reindex(df.index.normalize()).values
def run(N,ATRM,RR,fee,lo,hi,use=True):
    H=df.h.rolling(N).max().shift().values;L=df.l.rolling(N).min().shift().values
    a=df.atr.values;e=df.ema.values;c=df.c.values;h=df.h.values;l=df.l.values;rl=df.rel.values;idx=df.index
    eq=1.0;pos=None;day=None;deq=1;halt=False;out=[];curve=np.empty(len(df))
    for i in range(len(df)):
        if idx[i].date()!=day: day=idx[i].date();deq=eq;halt=False
        if pos:
            d,en,sl,tp,rk,e0,sz=pos;ex=None
            if d==1: ex=sl if l[i]<=sl else tp if h[i]>=tp else None
            else: ex=sl if h[i]>=sl else tp if l[i]<=tp else None
            if ex:
                eq+=sz*e0*((d*(ex-en)-2*fee*en)/rk);out.append(sz);pos=None
        if eq<deq*(1-DAILY_STOP):halt=True
        if not pos and not halt and not np.isnan(a[i]) and not np.isnan(H[i]) and not np.isnan(rl[i]):
            sz=RISK
            if use:
                if rl[i]<lo: sz=0
                elif rl[i]>hi: sz=RISK/2
            d=1 if(c[i]>H[i] and c[i]>e[i]) else -1 if(c[i]<L[i] and c[i]<e[i]) else 0
            if d and sz>0:
                rk=ATRM*a[i];pos=(d,c[i],c[i]-d*rk,c[i]+d*RR*rk,rk,eq,sz)
        curve[i]=eq
    s=pd.Series(curve,index=idx);m=s.resample("30D").last().pct_change().dropna()
    return len(out),s.iloc[-1]-1,(s/s.cummax()-1).min(),m.min()
print("N ATRM RR fee | régime | trades ret maxDD pire30j")
for fee in (0.0005,0.0002):
  for N,A,R in itertools.product((24,48),(1.5,3),(1.5,3)):
    for use in (False,True):
      t,r,d,w=run(N,A,R,fee,0.8,1.5,use)
      print(f"{N} {A} {R} {fee} | {'ON ' if use else 'off'} | {t:4d} {r:+7.1%} {d:7.1%} {w:6.1%}")
