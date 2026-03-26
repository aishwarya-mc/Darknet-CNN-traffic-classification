import pandas as pd

df = pd.read_parquet("cicdarknet2020.parquet")
df.to_csv("output.csv", index=False)
