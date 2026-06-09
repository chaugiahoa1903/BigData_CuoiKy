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
from pyspark.ml.regression import (
    LinearRegression,
    RandomForestRegressor,
    GBTRegressor,
)
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

spark = SparkSession.builder \
    .appName("Rossmann_BigData_MLlib") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")
HDFS_BASE = "hdfs://localhost:9000/user/project/rossmann"
features_rdd = spark.sparkContext.textFile(f"{HDFS_BASE}/features.json")
features_str = "\n".join(features_rdd.collect())
features = json.loads(features_str)

print("Features:", len(features))
sdf = spark.read.csv(
    f"{HDFS_BASE}/df_features.csv",
    header=True,
    inferSchema=True
)
print("Loaded df_features:", sdf.count(), "rows,", len(sdf.columns), "cols")