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
