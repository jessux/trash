import pandas as pd, numpy as np
df=pd.read_csv("BTCUSDT_15m.csv",index_col=0,parse_dates=True)
d=df.c.diff().abs(); day=df.index.date
g=df.groupby(day)
D=pd.DataFrame({"o":g.o.first(),"c":g.c.last(),"h":g.h.max(),"l":g.l.min(),"path":d.groupby(day).sum()})
D["er"]=(D.c-D.o).abs()/D.path          # efficacité du jour (1=tendance pure)
D["rng"]=(D.h-D.l)/D.o                  # volatilité du jour
D["ret"]=D.c/D.o-1
# label ex-post
D["trend"]=(D.er>D.er.rolling(60).median())
D["hivol"]=(D.rng>D.rng.rolling(60).median())
# features connues à 00:00 (veille et avant)
F=pd.DataFrame({"er1":D.er.shift(1),"er3":D.er.rolling(3).mean().shift(1),"rng1":D.rng.shift(1),
 "rngrel":(D.rng/D.rng.rolling(30).mean()).shift(1),"absret1":D.ret.abs().shift(1)})
X=F.join(D[["trend","hivol","er","rng","ret"]]).dropna()
print("jours",len(X),"| %trend",X.trend.mean().round(2))
for f in F.columns: print(f"{f:8s} corr(er_jour)={X[f].corr(X.er):+.2f}  corr(rng_jour)={X[f].corr(X.rng):+.2f}")
# prédiction simple : split moitié/moitié
from numpy.linalg import lstsq
h=len(X)//2; tr,te=X.iloc[:h],X.iloc[h:]
for tgt in ["er","rng"]:
    A=np.c_[tr[F.columns],np.ones(len(tr))]; w=lstsq(A,tr[tgt],rcond=None)[0]
    p=np.c_[te[F.columns],np.ones(len(te))]@w
    print(f"{tgt}: corr prédit/réel hors-échantillon = {np.corrcoef(p,te[tgt])[0,1]:+.2f}")
