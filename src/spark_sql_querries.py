from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Rossmann_SparkSQL").config("spark.sql.shuffle.partitions",4).getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Đọc dữ liệu từ HDFS
HDFS_ROOT = "hdfs://localhost:9000/user/project/rossmann"

df_sales = spark.read.csv(f"{HDFS_ROOT}/sale.csv", header = True, inferSchema=True)
df_stores = spark.read.csv(f"{HDFS_ROOT}/store.csv", header=True, inferSchema=True)

df_sales.createOrReplaceTempView("sales")
df_stores.createOrReplaceTempView("stores")

df_sales.cache()
df_stores.cache()

print(f"Sales : {df_sales.count():,} rows x {len(df_sales.columns)} cols")
print(f"Stores : {df_stores.count():,} rows x {len(df_stores.columns)} cols")

#Query 1: Ta sử dụng GroupBY, Aggeration và Join để tìm Doanh thu trunh bình theo từng loại cửa hàng

spark.sql("""
    SELECT
          s.StoreType,
          COUNT(DISTINCT sa.Store) AS so_cua_hang,
          ROUND(AVG(sa.Sales),2) AS doanh_thu_tb,
          ROUND(SUM(sa.Sales),2) AS tong_doanh_thu,
          ROUND(AVG(sa.Customers),2 AS khach_hang_tb
    FROM sales sa
    JOIN stores s ON sa.Store=s.Store
    WHERE sa.Open = 1 AND sa.Sales > 0
    GROUP BY s.StoreType
    ORDER BY doanh_thu_tb DESC
""").show()

#Query 2: Ta sử dụng GroupBY và Aggeration để so sánh doanh thu có Promo và không có Promo

spark.sql("""
    SELECT
          Promo,
          COUNT(*) AS so_ngay_ghi_nhan,
          ROUND(AVG(Sales),2) AS doanh_thu_tb,
          ROUND(AVG(Customers),2) as khach_tb,
          ROUND(AVG(Sales/Customers),2) as doanh_thu_per_khach
    FROM sales
    WHERE Open = 1 AND Sales > 0 AND Customers > 0
    GROUP BY Promo
    ORDER BY Promo
""").show()

#Query 3: Ta sử dụng Time series và GroupBy để doanh thu trung bình theo từng tháng trong năm

spark.sql("""
    SELECT
          MONTH(Date) as thang,
          ROUND(AVG(Sales),2) as doanh_thu_tb,
          ROUND(SUM(Sales),2) as tong_doanh_thu,
          ROUND(AVG(Customers),2) AS khach_tb
    FROM sales
    WHERE Open = 1 AND Sales > 0
    GROUP BY MONTH(Date)
    ORDER BY thang
""").show(12)

#Query 4: Xếp hạng top 10 cửa hàng có doanh thu trung bình cao nhất

spark.sql("""
    SELECT *
    FROM (
        SELECT
            sa.Store,
            s.StoreType,
            ROUND(AVG(sa.Sales), 2)         AS doanh_thu_tb,
            ROUND(SUM(sa.Sales), 2)         AS tong_doanh_thu,
            RANK() OVER (
                PARTITION BY s.StoreType
                ORDER BY AVG(sa.Sales) DESC
            )                               AS xep_hang
        FROM sales sa
        JOIN stores s ON sa.Store = s.Store
        WHERE sa.Open = 1 AND sa.Sales > 0
        GROUP BY sa.Store, s.StoreType
    )
    WHERE xep_hang <= 3
    ORDER BY StoreType, xep_hang
""").show()


