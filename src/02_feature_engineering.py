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

lags = [1, 3, 7, 14]

for lag in lags:
    df[f"Sales_lag_{lag}"]     = df.groupby("Store")["Sales"].shift(lag)
    df[f"Customers_lag_{lag}"] = df.groupby("Store")["Customers"].shift(lag)

windows = [7, 14]
for w in windows:
    df[f"Sales_roll_mean_{w}"] = df.groupby("Store")["Sales"] \
        .transform(lambda x: x.shift(1).rolling(window=w).mean())
    df[f"Sales_roll_std_{w}"]  = df.groupby("Store")["Sales"] \
        .transform(lambda x: x.shift(1).rolling(window=w).std())
    df[f"Sales_roll_max_{w}"]  = df.groupby("Store")["Sales"] \
        .transform(lambda x: x.shift(1).rolling(window=w).max())

df["Promo_Weekend"] = df["Promo"] * df["IsWeekend"]
df["Promo_Month"]   = df["Promo"] * df["Month"]
df["Sales_per_Customer"] = df["Sales"] / df["Customers"].replace(0, np.nan)

df = pd.get_dummies(df, columns=["DayOfWeek"], prefix = "DOW")

shape_before = df.shape
df = df.dropna().reset_index(drop=True)
print(f"Shape trước khi xử lý: {shape_before}")
print(f"Shape sau khi xử lý: {df.shape}")

categorical_cols = ["StateHoliday", "StoreType", "Assortment"]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

df = df.sort_values(by="Date").reset_index(drop=True)
