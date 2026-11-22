import pandas as pd, numpy as np
FEE=0.0005
def load(sym):
    raw=pd.read_csv(f"{sym}_15m.csv",index_col=0,parse_dates=True)
    df=raw.resample("1h").agg(dict(o="first",h="max",l="min",c="last")).dropna()
    tr=pd.concat([df.h-df.l,(df.h-df.c.shift()).abs(),(df.l-df.c.shift()).abs()],axis=1).max(axis=1)
    df["atr"]=tr.rolling(14).mean()
    dd=raw.resample("1D").agg(dict(o="first",h="max",l="min"));rng=(dd.h-dd.l)/dd.o
    df["rel"]=(rng/rng.rolling(30).mean()).shift(1).reindex(df.index.normalize()).values
    return df
def rmult(df,P=72,K=2,S=3,fee=FEE,lo=0.8,hi=1.2):
    """P&L par heure en multiples de R (1R = risque par trade)"""
    m=df.c.rolling(P).mean();s=df.c.rolling(P).std();up=(m+K*s).values;dn=(m-K*s).values;mid=m.values
    a=df.atr.values;c=df.c.values;h=df.h.values;l=df.l.values;rl=df.rel.values
    out=np.zeros(len(df));pos=None;n=0
    for i in range(len(df)):
        if pos:
            d,en,sl,rk,t0=pos;ex=None
            if d==1: ex=sl if l[i]<=sl else mid[i] if h[i]>=mid[i] else None
            else: ex=sl if h[i]>=sl else mid[i] if l[i]<=mid[i] else None
            if ex is None and i-t0>=48: ex=c[i]
            if ex is not None: out[i]=(d*(ex-en)-2*fee*en)/rk;n+=1;pos=None
        if not pos and not np.isnan(up[i]) and not np.isnan(a[i]) and not np.isnan(rl[i]) and lo<=rl[i]<=hi:
            d=1 if c[i]<dn[i] else -1 if c[i]>up[i] else 0
            if d: rk=S*a[i];pos=(d,c[i],c[i]-d*rk,rk,i)
    return pd.Series(out,index=df.index)
def challenge(daily,risk,target,dlim=0.05,mlim=0.10,mindays=4,maxdays=150):
    """daily: série P&L journalier en %capital initial. Retourne 'pass','fail' ou 'open'"""
    eq=0.0
    for k,(day,p) in enumerate(daily.items()):
        if k>=maxdays: return "open",k
        if p<=-dlim or eq+p<=-mlim: return "fail",k
        eq+=p
        if eq>=target and k+1>=mindays: return "pass",k+1
    return "open",len(daily)
if __name__=="__main__":
    syms=["BTCUSDT","ETHUSDT","SOLUSDT"];R={s:rmult(load(s)) for s in syms}
    for s in syms:
        r=R[s];print(s,"trades",(r!=0).sum(),"expectancy R/trade %.3f"%(r[r!=0].mean()),"total R %.1f"%r.sum())
    P=pd.concat(R,axis=1).sum(axis=1);print("portefeuille total R %.1f | trades %d"%(P.sum(),(pd.concat(R,axis=1)!=0).sum().sum()))
    for name,ser in [("BTC seul",R["BTCUSDT"]),("BTC+ETH+SOL",P)]:
      d=ser.resample("1D").sum()
      print("\n",name)
      for risk in (0.005,0.01,0.015):
        pd_=d*risk; res={"pass":0,"fail":0,"open":0}; n=0
        for st in range(0,len(pd_)-60,5):
            r_,_=challenge(pd_.iloc[st:],risk,0.10);res[r_]+=1;n+=1
        # phase 2 (5%) idem pas calculé séparément
        print(f"  risque {risk:.1%}/trade: phase1 (10%) réussite {res['pass']/n:.0%} | échec {res['fail']/n:.0%} | en cours {res['open']/n:.0%}  ({n} départs)")
