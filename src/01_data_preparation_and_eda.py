import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from math import sqrt

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler

from pyspark.ml.regression import (
    LinearRegression,
    RandomForestRegressor,
    GBTRegressor,
)
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

spark = SparkSession.builder \
        .appName("Rossmann_BigData_MLlib") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("Spark version:", spark.version)

HDFS_ROOT = "hdfs://localhost:9000/user/project/rossmann/"

df_sales_spark = spark.read.csv(f"{HDFS_ROOT}/sale.csv", header=True, inferSchema=True)
df_store_spark = spark.read.csv(f"{HDFS_ROOT}/store.csv", header=True, inferSchema=True)

print("Sales shape:", (df_sales_spark.count(), len(df_sales_spark.columns)))
print("Store shape:", (df_store_spark.count(), len(df_store_spark.columns)))

df_sales  = df_sales_spark.toPandas()
df_store  = df_store_spark.toPandas()

df_sales["Date"] = pd.to_datetime(df_sales["Date"])
df_sales = df_sales.sort_values("Date").reset_index(drop=True)

print("Sales shape:", df_sales.shape)
df_sales.head()

df_sales.describe()

print("Missing values – Sales:")
print(df_sales.isnull().sum())

print("Store shape:", df_store.shape)
df_store.head()
df_store.describe()
print("Missing values – Store:")
print(df_store.isnull().sum())

df_store["CompetitionDistance"] = df_store["CompetitionDistance"].fillna(df_store["CompetitionDistance"].mean(), inplace=True)

df_store.head()

df = df_sales.merge(right=df_store, on="Store", how="left")
df["Date"] = pd.to_datetime(df["Date"])
print("Merged shape:", df.shape)
df.head()

df_open = df[df["Sales"] > 0].copy()

plt.figure(figsize=(12, 8))
plt.hist(df_open["Sales"], bins=50)
plt.title("Phân phối của Sales")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.show()
#giá trị Sales daily
daily  = df_open.groupby("Date")["Sales"].mean()
promo_days = df_open[df_open["Promo"] == 1].groupby("Date")["Sales"].mean()

plt.figure(figsize=(14, 4))
plt.plot(daily.index, daily.values, color= "steelblue", linewidth=1)
plt.scatter(promo_days.index, promo_days.values, color="red", label="Có Promo", s=5)
plt.title("Doanh số trung bình Sales theo ngày", weight="bold")
plt.legend()
plt.show()

fig, axes = plt.subplots(1, 2 , figsize=(12, 4))

dow = df_open.groupby("DayOfWeek")["Sales"].mean()
axes[0].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], dow.values, color="steelblue")
axes[0].set_title("Sales theo ngày trong tuần", weight="bold")
axes[0].set_ylabel("Giá trị Sale trung bình", weight="bold")

month = df_open.groupby("DayOfWeeks")["Sales"].mean()
axes[1].bar(range(1, 13), month.values, color="green")
axes[1].set_title("Sales theo tháng", weight="bold")
axes[1].set_xlabel("Tháng", weight="bold")

plt.show()
#Tương quan biến
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Ma trận tương quan giữa các biến", weight="bold")
plt.show()

print(corr["Sales"].drop("Sales").sort_values(ascending=False))

#Xuất file đã qua xử lý
OUTPUT_PATH = f"{HDFS_ROOT}/processed_data.csv"
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print(f"Đã xuất file: {OUTPUT_PATH}")
print(f"   Shape: {df.shape}")