import requests, pandas as pd, time, sys
sym = sys.argv[1] if len(sys.argv)>1 else "BTCUSDT"
end = int(time.time()*1000); start = end - 730*86400*1000
rows=[]; t=start
while t<end:
    r=requests.get("https://data-api.binance.vision/api/v3/klines",params=dict(symbol=sym,interval="15m",startTime=t,limit=1000),timeout=20).json()
    if not r: break
    rows+=r; t=r[-1][0]+1
df=pd.DataFrame(rows).iloc[:,:6]; df.columns=["t","o","h","l","c","v"]
df["t"]=pd.to_datetime(df.t,unit="ms"); df=df.set_index("t").astype(float)
df.to_csv(f"{sym}_15m.csv"); print(len(df))
