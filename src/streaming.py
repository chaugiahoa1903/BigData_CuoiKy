# Import các thư viện cần thiết

import time
import threading
import math
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    IntegerType, FloatType, StringType, DateType
)

# Cấu hình cho hệ thống

HDFS_BASE        = "hdfs://localhost:9000/user/project/rossmann"
SOURCE_CSV       = f"{HDFS_BASE}/cleaned_rossmann.csv"
STREAM_INPUT_DIR = f"{HDFS_BASE}/stream_input"       # Producer ghi vào đây
STREAM_OUTPUT_DIR= f"{HDFS_BASE}/stream_output"      
CHECKPOINT_DIR   = f"{HDFS_BASE}/stream_checkpoint"  # Spark checkpoint

ANOMALY_DIR      = f"{HDFS_BASE}/stream_anomalies"   # Nơi ghi nhận các cảnh báo bất thường

ROWS_PER_BATCH   = 500      # Số dòng mỗi file batch
PRODUCER_INTERVAL= 4        # Giây giữa mỗi lần producer ghi file mới
TRIGGER_INTERVAL = "5 seconds"  # Spark sẽ xử lý mỗi 5 giây
MAX_BATCHES      = 15       # Dừng sau bao nhiêu batch

# Ngưỡng phát hiện bất thường theo quy tắc 3-sigma (z-score):
# z = (Sales - mean_lich_su) / std_lich_su của chính cửa hàng đó.
# |z| > 3  ->  chỉ xấp xỉ 0.3% dữ liệu bình thường bị rơi ra ngoài (quy tac 68-95-99.7)
# -> giá trị nằm ngoài khoảng này được xem là bất thường.
Z_THRESHOLD      = 2.5

# Mau ANSI cho terminal (do = sut giam, xanh la = tang vot)
import os as _os
_os.system("")          # bật ANSI color trên Windows PowerShell / CMD
RED    = "\033[91m"     # Sụt giảm bất thường
GREEN  = "\033[92m"     # Tăng đột biến
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# Khai báo Schema

ROSSMANN_SCHEMA = StructType([
    StructField("Store",         IntegerType(), True),
    StructField("DayOfWeek",     IntegerType(), True),
    StructField("Date",          StringType(),  True),  # đọc string, parse sau
    StructField("Sales",         FloatType(),   True),
    StructField("Customers",     IntegerType(), True),
    StructField("Open",          IntegerType(), True),
    StructField("Promo",         IntegerType(), True),
    StructField("StateHoliday",  StringType(),  True),
    StructField("SchoolHoliday", IntegerType(), True),
    StructField("StoreType",     StringType(),  True),
    StructField("Assortment",    StringType(),  True),
    StructField("CompetitionDistance", FloatType(), True),
    StructField("Year",          IntegerType(), True),
    StructField("Month",         IntegerType(), True),
    StructField("WeekOfYear",    IntegerType(), True),
    StructField("IsWeekend",     IntegerType(), True),
])

# Khởi tạo SparkSession

def create_spark_session() -> SparkSession:
    spark = (
        SparkSession.builder
        .appName("Rossmann_Structured_Streaming")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.adaptive.enabled", "false")
        # Tăng stack size JVM (-Xss) để tránh StackOverflowError do regex
        # Đệ quy sau khi Spark don dep checkpoint lúc dùng streaming query.
        .config("spark.driver.extraJavaOptions", "-Xss16m")
        .config("spark.executor.extraJavaOptions", "-Xss16m")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print("SparkSession khởi động thành công")
    print(f"Spark version : {spark.version}")
    print(f"Master        : {spark.sparkContext.master}")
    return spark