import numpy as np
import pandas as pd

df = pd.read_csv("../data/df_clean.csv")
df["Date"] = pd.to_datetime(df["Date"])
print("df_clean:", df.shape)
