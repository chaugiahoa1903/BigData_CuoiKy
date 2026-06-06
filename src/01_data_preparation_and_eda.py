import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.spark_session import get_spark

HDFS_ROOT   = "hdfs://localhost:9000/rossmann"
OUTPUT_PATH = f"{HDFS_ROOT}/clean/df_clean.parquet"

#Tải dữ liệu về từ HDFS rồi tạo DataFrame để xử lí
def load_data(spark):
    
    df_sales_spark  = spark.read.csv(f"{HDFS_ROOT}/sale.csv",
                                     header=True, inferSchema=True)
    df_stores_spark = spark.read.csv(f"{HDFS_ROOT}/store.csv",
                                     header=True, inferSchema=True)

    print(f"Sales  - {df_sales_spark.count():,} rows x {len(df_sales_spark.columns)} cols")
    print(f"Stores - {df_stores_spark.count():,} rows x {len(df_stores_spark.columns)} cols")

    df_sales  = df_sales_spark.toPandas()
    df_stores = df_stores_spark.toPandas()
    return df_sales, df_stores
