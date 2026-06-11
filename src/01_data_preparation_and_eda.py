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