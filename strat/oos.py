import itertools, pandas as pd, meanrev as M
full=M.df; mid=full.index[len(full)//2]
tr,te=full[full.index<mid],full[full.index>=mid]
res=[]
for fee in (0.0005,0.0002):
  for P,K,S in itertools.product((24,48,72),(2,2.5),(2,3)):
    M.df=tr; a=M.run(P,K,S,fee,0.8,1.2,True)
    M.df=te; b=M.run(P,K,S,fee,0.8,1.2,True)
    res.append((fee,P,K,S,a,b))
print("fee P K SLM | TRAIN(trades ret maxDD) | TEST(trades ret maxDD)")
for fee,P,K,S,a,b in sorted(res,key=lambda r:(r[0],-r[4][1])):
    print(f"{fee} {P} {K} {S} | {a[0]:3d} {a[1]:+6.1%} {a[2]:6.1%} | {b[0]:3d} {b[1]:+6.1%} {b[2]:6.1%}")
for fee in (0.0005,0.0002):
    r=[x for x in res if x[0]==fee]; best=max(r,key=lambda x:x[4][1])
    print(f"fee {fee}: meilleur train = P{best[1]} K{best[2]} S{best[3]} -> test {best[5][1]:+.1%}; moyenne test des 12 configs {sum(x[5][1] for x in r)/len(r):+.1%}; positives {sum(x[5][1]>0 for x in r)}/12")
