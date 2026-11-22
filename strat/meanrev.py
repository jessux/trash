import pandas as pd, numpy as np, itertools
RISK=0.005; DAILY_STOP=0.03
raw=pd.read_csv("BTCUSDT_15m.csv",index_col=0,parse_dates=True)
df=raw.resample("1h").agg(dict(o="first",h="max",l="min",c="last")).dropna()
tr=pd.concat([df.h-df.l,(df.h-df.c.shift()).abs(),(df.l-df.c.shift()).abs()],axis=1).max(axis=1)
df["atr"]=tr.rolling(14).mean()
dd=raw.resample("1D").agg(dict(o="first",h="max",l="min"));rng=(dd.h-dd.l)/dd.o
df["rel"]=(rng/rng.rolling(30).mean()).shift(1).reindex(df.index.normalize()).values
def run(P,K,SLM,fee,lo,hi,use):
    m=df.c.rolling(P).mean();s=df.c.rolling(P).std();up=(m+K*s).values;dn=(m-K*s).values;mid=m.values
    a=df.atr.values;c=df.c.values;h=df.h.values;l=df.l.values;rl=df.rel.values;idx=df.index
    eq=1.0;pos=None;day=None;deq=1;halt=False;n=0;curve=np.empty(len(df))
    for i in range(len(df)):
        if idx[i].date()!=day:day=idx[i].date();deq=eq;halt=False
        if pos:
            d,en,sl,rk,sz,t0=pos;ex=None
            if d==1: ex=sl if l[i]<=sl else mid[i] if h[i]>=mid[i] else None
            else: ex=sl if h[i]>=sl else mid[i] if l[i]<=mid[i] else None
            if ex is None and i-t0>=48: ex=c[i]
            if ex is not None:
                eq+=sz*((d*(ex-en)-2*fee*en)/rk);n+=1;pos=None
        if eq<deq*(1-DAILY_STOP):halt=True
        if not pos and not halt and not np.isnan(up[i]) and not np.isnan(a[i]) and not np.isnan(rl[i]):
            sz=RISK
            if use:
                if rl[i]<lo or rl[i]>hi: sz=0
            d=1 if c[i]<dn[i] else -1 if c[i]>up[i] else 0
            if d and sz>0:
                rk=SLM*a[i];pos=(d,c[i],c[i]-d*rk,rk,sz,i)
        curve[i]=eq
    s_=pd.Series(curve,index=idx);mm=s_.resample("30D").last().pct_change().dropna()
    return n,s_.iloc[-1]-1,(s_/s_.cummax()-1).min(),mm.min()
if __name__=="__main__":
    print("P K SLM fee | filtre | trades ret maxDD pire30j")
    for fee in (0.0005,0.0002):
      for P,K,S in itertools.product((24,48),(2,2.5),(2,3)):
        for use in (False,True):
          t,r,d,w=run(P,K,S,fee,0.8,1.2,use)
          print(f"{P} {K} {S} {fee} | {'ON ' if use else 'off'} | {t:4d} {r:+7.1%} {d:7.1%} {w:6.1%}")
