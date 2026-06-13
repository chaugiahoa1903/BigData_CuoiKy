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

# Khởi tạo Producer - Chạy trong Thread riêng

def write_single_csv_hdfs(spark, sdf, hdfs_file_path):
    """
    Ghi sdf thành 1 file CSV PHẲNG trên HDFS
    Cách làm: Ta sẽ ghi ra folder tạm  tìm part-file  rename thành file phẳng
    Đây cũng là pattern chuẩn cho file streaming source:
    File chỉ xuất hiện trong thư mục theo dõi sau khi đã ghi xong hoàn toàn,
    tránh việc Spark đọc phải file đang ghi dở.
    """
    # Bước 1: Ghi ra folder tạm (coalesce(1) gop thanh 1 part-file duy nhat)
    tmp = hdfs_file_path + "__tmp"
    (sdf.coalesce(1)
        .write.mode("overwrite")
        .option("header", "true")
        .csv(tmp))
 
    # Bước 2: Dùng Hadoop FileSystem API (qua JVM) để thao tác với file trên HDFS
    hadoop = spark._jvm.org.apache.hadoop
    conf   = spark._jsc.hadoopConfiguration()
    uri    = spark._jvm.java.net.URI(tmp)
    fs     = hadoop.fs.FileSystem.get(uri, conf)
    Path   = hadoop.fs.Path
 
    # Tìm file part-xxxxx.csv trong folder tạm
    part = None
    for st in fs.listStatus(Path(tmp)):
        nm = str(st.getPath().getName())
        if nm.startswith("part-") and nm.endswith(".csv"):
            part = st.getPath()
            break
 
    dst = Path(hdfs_file_path)
    if fs.exists(dst):
        fs.delete(dst, False)
    fs.rename(part, dst)          # rename part-file thành file phẳng
    fs.delete(Path(tmp), True)    # xoá folder tạm
 
 
def run_producer(spark: SparkSession, stop_event: threading.Event):
    """
    Producer giúp ta giả lập các cửa hàng Rossmann gửi dữ liệu lên HDFS theo thời gian thực.
    Mỗi PRODUCER_INTERVAL giây, một file CSV nhỏ với khoảng 500 dòng sẽ được ghi vào
    stream_input/, kích hoạt Spark Structured Streaming xử lý micro-batch mới.
    """
    # Đọc toàn bộ dataset gốc từ HDFS 1 lần duy nhất
    print("\nProducer: Đang đọc toàn bộ dataset từ HDFS...")
 
    try:
        df_full = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(SOURCE_CSV)
        )
        total_rows = df_full.count()
        print(f"Tổng số dòng dataset: {total_rows:,}")
    except Exception as e:
        print(f"Producer: Không đọc được CSV từ HDFS — {e}")
        stop_event.set()
        return
 
    # Tính số batch cần thiết
    n_batches = min(MAX_BATCHES, math.ceil(total_rows / ROWS_PER_BATCH))
    print(f"Sẽ gửi {n_batches} batch × {ROWS_PER_BATCH} dòng/batch")
    print(f"Ghi vào: {STREAM_INPUT_DIR}/\n")
 
    # Lấy về Pandas để dễ slice theo batch
    pdf = df_full.toPandas()
 
    # Lặp lại qua từng batch: lấy ra 500 dòng rồi ghi thành 1 file -> gửi file dần dần
    for batch_idx in range(n_batches):
        if stop_event.is_set():
            break
 
        start_row = batch_idx * ROWS_PER_BATCH
        end_row   = min(start_row + ROWS_PER_BATCH, total_rows)
        batch_pdf = pdf.iloc[start_row:end_row]
 
        # Ghi batch lên HDFS dưới dạng FILE PHẲNG, để Spark readStream có thể xử lý được
        output_path = f"{STREAM_INPUT_DIR}/batch_{batch_idx:04d}.csv"
        batch_sdf = spark.createDataFrame(batch_pdf)
        write_single_csv_hdfs(spark, batch_sdf, output_path)
 
        ts = time.strftime("%H:%M:%S")
        print(
            f"  [{ts}]  Producer gửi batch {batch_idx+1:02d}/{n_batches} "
            f"— dòng {start_row:,}{end_row:,} "
            f"({end_row - start_row} records)"
        )
 
        time.sleep(PRODUCER_INTERVAL)
 
    print("\nProducer: Đã gửi xong tất cả các batch.")
    stop_event.set()