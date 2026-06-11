import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, lit

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

spark = (
    SparkSession.builder
    .appName("Rossmann_03_MLlib")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

HDFS_ROOT = "hdfs://localhost:9000/user/project/rossmann"

features_str = "\n".join(spark.sparkContext.textFile(f"{HDFS_ROOT}/features.json").collect())
features = json.loads(features_str)
print("Features:", len(features))

sdf = spark.read.csv(f"{HDFS_ROOT}/df_features.csv", header=True, inferSchema=True)
print("Loaded df_features:", sdf.count(), "rows,", len(sdf.columns), "cols")

total       = sdf.count()
train_count = int(total * 0.8)

w   = Window.orderBy(lit(1))
sdf = sdf.withColumn("_row_id", row_number().over(w))

train_sdf = sdf.filter(F.col("_row_id") <= train_count).drop("_row_id")
test_sdf  = sdf.filter(F.col("_row_id") >  train_count).drop("_row_id")

train_sdf.cache()
test_sdf.cache()
print(f"Số mẫu train : {train_sdf.count():,}")
print(f"Số mẫu test  : {test_sdf.count():,}")