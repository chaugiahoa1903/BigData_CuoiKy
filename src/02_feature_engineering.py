import json

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, UnivariateFeatureSelector

spark = (
    SparkSession.builder
    .appName("Rossmann_02_FeatureEngineering_Spark")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
HDFS_ROOT = "hdfs://localhost:9000/user/project/rossmann"

def save_single_csv(sdf, hdfs_path):
    tmp = hdfs_path + "__tmp"
    (sdf.coalesce(1)
        .write.mode("overwrite")
        .option("header", "true")
        .csv(tmp))
    
    hadoop = spark._jvm.org.apache.hadoop
    conf = spark._jsc.hadoopConfiguration()
    
    # Fix: dùng URI để buộc lấy HDFS filesystem
    uri = spark._jvm.java.net.URI(tmp)
    fs = hadoop.fs.FileSystem.get(uri, conf)  # ← dòng fix
    
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


def save_json_hdfs(obj, hdfs_path):
    content = json.dumps(obj, ensure_ascii=False, indent=2)
    hadoop = spark._jvm.org.apache.hadoop
    fs = hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
    out = fs.create(hadoop.fs.Path(hdfs_path), True)
    out.write(bytearray(content, "utf-8"))
    out.close()
    print("Da ghi:", hdfs_path)


def add_dummies(df, col, drop_first=True):
    cats = [r[0] for r in df.select(col).distinct().collect() if r[0] is not None]
    cats = sorted(cats, key=lambda x: str(x))
    if drop_first:
        cats = cats[1:]
    new_cols = []
    for c in cats:
        safe = f"{col}_{str(c).replace(' ', '_')}"
        df = df.withColumn(safe, F.when(F.col(col) == c, 1).otherwise(0))
        new_cols.append(safe)
    return df, new_cols

