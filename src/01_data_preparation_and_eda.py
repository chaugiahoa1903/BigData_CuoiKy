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

#Thực hiện EDA 


def run_eda(df: pd.DataFrame) -> None:
    df_open = df[df["Sales"] > 0].copy()

#Phân phối của Sales
    plt.figure(figsize=(12, 8))
    plt.hist(df_open["Sales"], bins=50)
    plt.title("Phân phối của Sales")
    plt.xlabel("Sales")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


#Giá trị trung bình của Sales theo ngày và phân biệt giữa ngày có Promo và không có Promo
    daily      = df_open.groupby("Date")["Sales"].mean()
    promo_days = df_open[df_open["Promo"] == 1].groupby("Date")["Sales"].mean()

    plt.figure(figsize=(14, 4))
    plt.plot(daily.index, daily.values, color="steelblue", linewidth=1)
    plt.scatter(promo_days.index, promo_days.values,
                color="red", s=5, label="Có Promo")
    plt.title("Giá trị trung bình của Sales theo ngày", weight="bold")
    plt.legend()
    plt.tight_layout()
    plt.show()


#Sales theo Ngày trong Tuần, theo Tháng
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    dow = df_open.groupby("DayOfWeek")["Sales"].mean()
    axes[0].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                dow.values, color="steelblue")
    axes[0].set_title("Sales theo Ngày trong Tuần", weight="bold")
    axes[0].set_ylabel("Giá trị Sale trung bình", weight="bold")

    month = df_open.groupby(df_open["Date"].dt.month)["Sales"].mean()
    axes[1].bar(range(1, 13), month.values, color="green")
    axes[1].set_title("Sales theo Tháng", weight="bold")
    axes[1].set_xlabel("Tháng", weight="bold")

    plt.tight_layout()
    plt.show()

#Ma trận tương quan giữa các biến và top tương quan với Sales
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(14, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Ma trận tương quan giữa các biến", weight="bold")
    plt.tight_layout()
    plt.show()

    print("\nTop tương quan với Sales:")
    print(corr["Sales"].drop("Sales").sort_values(ascending=False))

#Xuất DataFrame đã sạch ra HDFS dưới dạng Parquet

def export_clean(df: pd.DataFrame, spark) -> None:
    
    df_export = df.copy()
    df_export["Date"] = df_export["Date"].astype(str)
    for col in df_export.select_dtypes(include=[object]).columns:
        df_export[col] = df_export[col].astype(str)
    sdf = spark.createDataFrame(df_export)
    sdf.write.mode("overwrite").parquet(OUTPUT_PATH)
    print(f"\n Đã lưu df_clean → {OUTPUT_PATH}")

#Chạy toàn bộ pipeline

if __name__ == "__main__":
    spark = get_spark("01_Data_EDA")

    df_sales, df_stores = load_data(spark)
    df = clean_and_join(df_sales, df_stores)
    run_eda(df)
    export_clean(df, spark)

    spark.stop()
    print("\n[01] Done.")