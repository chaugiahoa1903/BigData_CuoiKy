import numpy as np
import pandas as pd

df = pd.read_csv("../data/df_clean.csv")
df["Date"] = pd.to_datetime(df["Date"])
print("df_clean:", df.shape)

df ["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(by=["Store","Date"]).reset_index(drop=True)

df["Year"]      = df["Date"].dt.year
df["Month"]     = df["Date"].dt.month
df["WeekOfYear"]= df["Date"].dt.isocalendar().week.astype(int)
df["DayOfWeek"] = df["Date"].dt.dayofweek + 1
df["IsWeekend"] = df["DayOfWeek"].isin([6, 7]).astype(int)

