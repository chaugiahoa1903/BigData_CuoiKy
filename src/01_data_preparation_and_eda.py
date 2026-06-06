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
# Làm sạch dữ liệu và join stores vào sales
#Ở CompetitionDistance thay thế giá trị thiếu bằng giá trị trung bình.
#Ở Promo2 thay thế giá trị thiếu bằng 0.
#chuyển cột Date thành định dạng datetime và join DataFrame dựa trên cột Store.
def clean_and_join(df_sales: pd.DataFrame,
                   df_stores: pd.DataFrame) -> pd.DataFrame:
    
    print("\n── Bộ dữ liệu Sales ──")
    print(df_sales.shape)
    print(df_sales.isnull().sum())

    print("\n── Bộ dữ liệu Stores ──")
    print(df_stores.shape)
    print(df_stores.isnull().sum())

    df_stores["CompetitionDistance"] = df_stores[
        "CompetitionDistance"
    ].fillna(df_stores["CompetitionDistance"].mean())


    df_stores["Promo2"] = df_stores["Promo2"].fillna(0)

   
    df_sales["Date"] = pd.to_datetime(df_sales["Date"])
    df = df_sales.merge(right=df_stores, on="Store", how="left")
    df["Date"] = pd.to_datetime(df["Date"])

    print(f"\n Joined shape: {df.shape}")
    print(df.head())
    return df