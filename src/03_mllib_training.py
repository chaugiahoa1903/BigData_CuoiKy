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

total = sdf.count()
train_count = int(total * 0.8)
w   = Window.orderBy(lit(1))
sdf = sdf.withColumn("_row_id", row_number().over(w))

train_sdf = sdf.filter(F.col("_row_id") <= train_count).drop("_row_id")
test_sdf  = sdf.filter(F.col("_row_id") >  train_count).drop("_row_id")

train_sdf.cache()
test_sdf.cache()

print(f"Số mẫu train : {train_sdf.count():,}")
print(f"Số mẫu test  : {test_sdf.count():,}")

assembler = VectorAssembler(
    inputCols=features,
    outputCol="features_vec",
    handleInvalid="skip"
)

scaler = StandardScaler(
    inputCol="features_vec",
    outputCol="features_scaled",
    withStd=True,
    withMean=False
)

evaluator_rmse = RegressionEvaluator(
    labelCol="Sales", predictionCol="prediction", metricName="rmse"
)
evaluator_mae = RegressionEvaluator(
    labelCol="Sales", predictionCol="prediction", metricName="mae"
)
evaluator_r2 = RegressionEvaluator(
    labelCol="Sales", predictionCol="prediction", metricName="r2"
)
tune_sdf, _ = train_sdf.randomSplit([0.1, 0.9], seed=42)
tune_sdf.cache()

print(f"Số mẫu tune  : {tune_sdf.count():,}  (10% train)")
print(f"\n{'='*55}")
print("  Tuning: RandomForestRegressor")
print(f"{'='*55}")

rf_tune = RandomForestRegressor(
    featuresCol="features_scaled",
    labelCol="Sales",
    seed=42
)

pipeline_rf_tune = Pipeline(stages=[assembler, scaler, rf_tune])

rf_param_grid = (
    ParamGridBuilder()
    .addGrid(rf_tune.numTrees,            [50, 100])
    .addGrid(rf_tune.maxDepth,            [6, 8])
    .addGrid(rf_tune.minInstancesPerNode, [2])
    .build()
)
cv_rf = CrossValidator(
    estimator=pipeline_rf_tune,
    estimatorParamMaps=rf_param_grid,
    evaluator=evaluator_rmse,
    numFolds=2,
    seed=42,
    parallelism=1   
)

print("Đang chạy CrossValidator RF (4 tổ hợp × 2 folds = 8 fits)...")
cv_rf_model = cv_rf.fit(tune_sdf)

best_rf_stage    = cv_rf_model.bestModel.stages[-1]
best_rf_numTrees = best_rf_stage.getNumTrees
best_rf_maxDepth = best_rf_stage.getMaxDepth()
best_rf_minInst  = best_rf_stage.getMinInstancesPerNode()

print(f"  RF best numTrees          : {best_rf_numTrees}")
print(f"  RF best maxDepth          : {best_rf_maxDepth}")
print(f"  RF best minInstancesPerNode: {best_rf_minInst}")

print(f"\n{'='*55}")
print("  Tuning: GBTRegressor")
print(f"{'='*55}")

gbt_tune = GBTRegressor(
    featuresCol="features_scaled",
    labelCol="Sales",
    seed=42
)

pipeline_gbt_tune = Pipeline(stages=[assembler, scaler, gbt_tune])

gbt_param_grid = (
    ParamGridBuilder()
    .addGrid(gbt_tune.maxIter,   [50, 100])  # bỏ stepSize nhỏ, giữ 2 giá trị quan trọng nhất
    .addGrid(gbt_tune.maxDepth,  [4, 6])
    .addGrid(gbt_tune.stepSize,  [0.1])      # cố định 1 giá trị
    .build()
)

cv_gbt = CrossValidator(
    estimator=pipeline_gbt_tune,
    estimatorParamMaps=gbt_param_grid,
    evaluator=evaluator_rmse,
    numFolds=2,
    seed=42,
    parallelism=1
)

print("Đang chạy CrossValidator GBT (4 tổ hợp × 2 folds = 8 fits)...")
cv_gbt_model = cv_gbt.fit(tune_sdf)

best_gbt_stage    = cv_gbt_model.bestModel.stages[-1]
best_gbt_maxIter  = best_gbt_stage.getMaxIter()
best_gbt_maxDepth = best_gbt_stage.getMaxDepth()
best_gbt_stepSize = best_gbt_stage.getStepSize()

print(f"GBT best maxIter  : {best_gbt_maxIter}")
print(f"GBT best maxDepth : {best_gbt_maxDepth}")
print(f"GBT best stepSize : {best_gbt_stepSize}")

lr = LinearRegression(
    featuresCol="features_scaled",
    labelCol="Sales",
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.0
)

rf = RandomForestRegressor(
    featuresCol="features_scaled",
    labelCol="Sales",
    numTrees=best_rf_numTrees,
    maxDepth=best_rf_maxDepth,
    minInstancesPerNode=best_rf_minInst,
    seed=42
)

gbt = GBTRegressor(
    featuresCol="features_scaled",
    labelCol="Sales",
    maxIter=best_gbt_maxIter,
    maxDepth=best_gbt_maxDepth,
    stepSize=best_gbt_stepSize,
    subsamplingRate=0.8,
    seed=42
)

model_configs = {
    "LinearRegression"    : lr,
    "RandomForestRegressor": rf,
    "GBTRegressor"        : gbt,
}