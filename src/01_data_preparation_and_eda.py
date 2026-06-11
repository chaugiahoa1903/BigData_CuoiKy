import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.functions import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

spark = (
    SparkSession.builder
    .appName("Rossmann_01_EDA_Spark").getOrCreate()
    .coonfig("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print("Spark version:", spark.version)
HDFS_ROOT = "hdfs://localhost:9000/user/user/rossmann/"
#Ghi 1 file csv ra hdfs
def save_single_csv(sdf, hdfs_path):
    tmp = hdfs_path + "_tmp"
    (sdf.coalesce(1)
        .write.mode("overwrite")
        .option("header", "true")
        .csv(tmp))
    
    hadoop = spark._jvm.org.apache.hadoop
    conf = spark._jsc.hadoopConfiguration()
    
    uri = spark._jvm.java.net.URI(tmp)
    fs = hadoop.fs.FileSystem.get(uri, conf)

    Path = hadoop.fs.Path
    part = None
    for st in fs.listStatus(Path(tmp)):
        nm = str(st.getPath().getName())
        if nm.startswith("part-") and nm.endswith(".csv"):
            part = st.getPath()
            break
    dst = Path(hdfs_path)
    if fs.exists(dst):
        fs.delete(dst, True)
    fs.rename(part, dst)
    fs.delete(Path(tmp), True)
    print("Da ghi:", hdfs_path)


#Đếm missing values
def show_null_counts(df, title):
    print(title)
    df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]).show()


#Tiền xử lý bộ dữ liệu sale
df_sales = spark.read.csv(f"{HDFS_ROOT}/sale.csv", header=True, inferSchema=True)
df_sales = df_sales.withColumn("Date", F.to_date("Date", "M/d/yyyy"))
df_sales = df_sales.orderBy("Date")

print("sale.csv shape :", (df_sales.count(), len(df_sales.columns)))   # ~ (1017209, 8)
df_sales.show(5)
df_sales.describe().show()

show_null_counts(df_sales, "Missing values - sale.csv:")


#Tiền xử lý bộ dữ liệu store
df_store = spark.read.csv(f"{HDFS_ROOT}/store.csv", header=True, inferSchema=True)

print("store.csv shape:", (df_store.count(), len(df_store.columns)))   # ~ (1115, 10)
df_store.show(5)
df_store.describe().show()

show_null_counts(df_store, "Missing values - store.csv (truoc xu ly):")

median_cd = df_store.approxQuantile("CompetitionDistance", [0.5], 0.001)[0]
print("Median CompetitionDistance:", median_cd)
df_store = df_store.na.fill({"CompetitionDistance": median_cd})

df_store = df_store.na.fill(0).na.fill("0")

show_null_counts(df_store, "Missing values - store.csv (sau xu ly):")
df_store.show(5)

#Kết hợp 2 bộ dữ liệu
df = df_sales.join(df_store, on="Store", how="left")
print("Merged shape:", (df.count(), len(df.columns))) 
df.show(5)
