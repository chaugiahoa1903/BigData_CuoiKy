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
#ghi 1 file csv ra hdfs
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
#đếm missing values
    def show_null_counts(df, title):
        print(title)
        df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]).show()