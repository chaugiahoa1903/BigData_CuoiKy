# Import các thư viện cần thiết
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import base64
import os
import json

#Khởi tạo hàm đọc file hình ảnh

@st.cache_data
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        st.error(f"Không tìm thấy tệp hình ảnh '{bin_file}'.")
        return ""
    
#Cấu hình trang: Tiêu đề, icon và layout toàn màn hình

st.set_page_config(
    page_title="Rossmann Forecaster",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="collapsed"
)

img_path = 'image_cacfcb.jpg'
img_base64 = get_base64_of_bin_file(img_path)

st.markdown(f"""
<style>
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    .block-container {{padding-top: 1rem; max-width: 85rem;}}

    .glass-card {{
        background-color: white;
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.05);
        border: 1px solid #f3f4f6;
        height: 100%;
    }}

    .hero-banner {{
        background: linear-gradient(90deg, rgba(227,6,19,0.9) 0%, rgba(227,6,19,0.3) 100%),
                    url('data:image/jpeg;base64,{img_base64}');
        background-size: cover;
        background-position: center 20%;
        border-radius: 1.5rem;
        padding: 4.5rem 2rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(227,6,19,0.3);
    }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 2rem; }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }}
    .stTabs [aria-selected="true"] {{
        color: #E30613 !important;
        border-bottom: 3px solid #E30613 !important;
        font-weight: bold;
    }}
</style>
""", unsafe_allow_html=True)

#Header: Tên thương hiệu + logo và thông tin người dùng

c1, c2 = st.columns([3, 1])
logo_path = 'logo.jpg'
logo_base64 = get_base64_of_bin_file(logo_path)

with c1:
    st.markdown(f"""
        <div style='display: flex; align-items: center; gap: 16px; padding-bottom: 15px;'>
            <div style='background-color: white; padding: 10px; border-radius: 12px;
                        border: 1px solid #E5E7EB; display: flex; justify-content: center;
                        align-items: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05);'>
                <img src="data:image/png;base64,{logo_base64}"
                     style="width: 48px; height: 48px; object-fit: contain;" alt="Rossmann Logo">
            </div>
            <div>
                <h1 style='color: #E30613; font-size: 28px; margin: 0;
                           font-weight: 800; letter-spacing: -0.5px;'>ROSSMANN</h1>
                <p style='color: #6B7280; font-size: 12px; margin: 0;
                          text-transform: uppercase; letter-spacing: 2px; font-weight: bold;'>
                    Sales Forecaster
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("""
        <div style='display: flex; align-items: center; justify-content: flex-end;
                    gap: 12px; padding-top: 10px;'>
            <div style='text-align: right;'>
                <p style='margin: 0; font-size: 16px; font-weight: 800; color: #000000 !important;
                          font-family: "Segoe UI", Roboto, Arial, sans-serif; letter-spacing: 0.5px;'>
                    Người Dùng
                </p>
                <p style='margin: 0; font-size: 12px; font-weight: 700; color: #047857 !important;
                          display: flex; align-items: center; justify-content: flex-end; gap: 5px;
                          letter-spacing: 0.5px; font-family: "Segoe UI", Roboto, Arial, sans-serif;'>
                    <span style='height: 8px; width: 8px; background-color: #10B981;
                                 border-radius: 50%; display: inline-block;
                                 box-shadow: 0 0 4px #10B981;'></span>
                    Đang hoạt động
                </p>
            </div>
            <div style='width: 42px; height: 42px; border-radius: 50%; background-color: #F3F4F6;
                        border: 1px solid #D1D5DB; display: flex; justify-content: center;
                        align-items: center; font-size: 20px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
                ?
            </div>
        </div>
    """, unsafe_allow_html=True)

# Cấu hình đường dẫn đọc file trên HDFS

_HDFS_ROOT          = "hdfs://localhost:9000/user/project/rossmann"
MLLIB_MODEL_PATH    = f"{_HDFS_ROOT}/models/GBTRegressor"
CLEANED_CSV_HDFS    = f"{_HDFS_ROOT}/cleaned_rossmann.csv"
FEATURES_JSON_HDFS  = f"{_HDFS_ROOT}/features.json"


def _find_hdfs_cmd():
    """
    Tìm đường dẫn đầy đủ tới lệnh hdfs.
    Trên Windows lệnh thật là hdfs.cmd (subprocess không tự thêm .cmd như
    PowerShell), nên thử lần lượt: hdfs.cmd / hdfs trong PATH, rồi
    %HADOOP_HOME%\\bin. Tránh lỗi 'Không tìm thấy lệnh hdfs'.
    """
    import shutil
    for name in ("hdfs.cmd", "hdfs"):
        found = shutil.which(name)
        if found:
            return found
    hadoop_home = os.environ.get("HADOOP_HOME")
    if hadoop_home:
        for name in ("hdfs.cmd", "hdfs"):
            cand = os.path.join(hadoop_home, "bin", name)
            if os.path.exists(cand):
                return cand
    return None

# Khởi động SparkSession và load model GBT đã train vào từ HDFS.
# @st.cache_resource: chỉ chạy 1 lần, giữ lại để rerun cho các lần sau.

@st.cache_resource(show_spinner="Đang khởi động Spark & load MLlib model...")
def load_model():
    try:
        from pyspark.ml import PipelineModel
        from pyspark.sql import SparkSession

        spark = (SparkSession.builder
                 .appName("RossmannWebapp")
                 .config("spark.sql.shuffle.partitions", "4")
                 .config("spark.driver.memory", "2g")
                 .getOrCreate())
        spark.sparkContext.setLogLevel("ERROR")
        model = PipelineModel.load(MLLIB_MODEL_PATH)
        return model, spark
    except Exception as e:
        st.error(f"Không load được MLlib model: {e}")
        return None, None

# Dự báo hàng loạt: pandas -> spark DataFrame -> model.transform -> Lấy cột prediction

def mllib_predict_batch(pipeline_model, spark, X_pandas, features):
    X_input = X_pandas[features].copy().astype("float64")
    sdf = spark.createDataFrame(X_input)
    preds_sdf = pipeline_model.transform(sdf)
    return preds_sdf.select("prediction").toPandas()["prediction"].values

# Nạp data từ hdfs để tạo các features (lag, rolling, Promo, one-hot)

@st.cache_data(show_spinner="Đang load dữ liệu từ HDFS...")
def load_and_preprocess_data():
    try:
        from pyspark.sql import SparkSession
        _spark = (SparkSession.builder
                  .appName("RossmannWebapp")
                  .config("spark.sql.shuffle.partitions", "4")
                  .config("spark.driver.memory", "2g")
                  .getOrCreate())
        _spark.sparkContext.setLogLevel("ERROR")
        # Đọc trực tiếp từ HDFS rồi chuyển sang pandas để xử lý phía sau
        df = (_spark.read
              .option("header", "true")
              .option("inferSchema", "true")
              .csv(CLEANED_CSV_HDFS)
              .toPandas())
    except Exception:
        # Fallback dữ liệu giả lập nếu không kết nối được HDFS
        dates = pd.date_range("2015-05-01", "2015-07-31")
        data = []
        for s in [1, 2, 3]:
            for d in dates:
                data.append([s, d, np.random.randint(3000, 8000),
                             np.random.randint(300, 800),
                             np.random.choice([0, 1]), 0, 0, 'a', 'a'])
        df = pd.DataFrame(data, columns=[
            'Store', 'Date', 'Sales', 'Customers', 'Promo',
            'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment'
        ])

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Store', 'Date']).reset_index(drop=True)

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['DayOfWeek'] = df['Date'].dt.dayofweek + 1
    df['IsWeekend'] = df['DayOfWeek'].isin([6, 7]).astype(int)

    lags = [1, 3, 7, 14]
    for lag in lags:
        df[f'Sales_lag_{lag}'] = df.groupby('Store')['Sales'].shift(lag)
        df[f'Customers_lag_{lag}'] = df.groupby('Store')['Customers'].shift(lag)

    windows = [7, 14]
    for w in windows:
        df[f'Sales_roll_mean_{w}'] = df.groupby('Store')['Sales'].transform(
            lambda x: x.shift(1).rolling(window=w).mean())
        df[f'Sales_roll_std_{w}'] = df.groupby('Store')['Sales'].transform(
            lambda x: x.shift(1).rolling(window=w).std())
        df[f'Sales_roll_max_{w}'] = df.groupby('Store')['Sales'].transform(
            lambda x: x.shift(1).rolling(window=w).max())

    df['Promo_Weekend'] = df['Promo'] * df['IsWeekend']
    df['Promo_Month'] = df['Promo'] * df['Month']
    df['Sales_per_Customer'] = np.where(
        df['Customers'] == 0, 0, df['Sales'] / df['Customers'])

    df = pd.get_dummies(df, columns=['DayOfWeek'], prefix='DOW')
    df = df.dropna(
        subset=[f'Sales_lag_{lag}' for lag in lags] +
               [f'Sales_roll_mean_{w}' for w in windows]
    ).reset_index(drop=True)

    categorical_cols = [c for c in ['StateHoliday', 'StoreType', 'Assortment']
                        if c in df.columns]
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df = df.sort_values(by='Date')

    exclude_cols = ['Date', 'Sales', 'Customers', 'PromoInterval', 'Store']
    features = [col for col in df.columns if col not in exclude_cols]
    return df, features

# Điền lag/rolling cho 1 ngày trong tương lai dựa trên các ngày đã dự báo

def fill_lag_rolling(df_ref, i, cust_dow_avg, hist_df_full):
    for lag in [1, 3, 7, 14]:
        idx = i - lag
        df_ref.loc[i, f"Sales_lag_{lag}"] = (
            df_ref.loc[idx, "Sales"] if idx >= 0 else np.nan)
        val = df_ref.loc[idx, "Customers"] if idx >= 0 else np.nan
        if pd.isna(val):
            val = cust_dow_avg.get(df_ref.loc[i, "Date"].dayofweek + 1, 0)
        df_ref.loc[i, f"Customers_lag_{lag}"] = val
 
    for w in [7, 14]:
        history = df_ref.loc[max(0, i - w): i - 1, "Sales"].dropna()
        df_ref.loc[i, f"Sales_roll_mean_{w}"] = (
            history.mean() if len(history) > 0 else np.nan)
        df_ref.loc[i, f"Sales_roll_std_{w}"] = (
            history.std() if len(history) > 1 else 0.0)
        df_ref.loc[i, f"Sales_roll_max_{w}"] = (
            history.max() if len(history) > 0 else np.nan)
 
    s_lag1 = df_ref.loc[i, "Sales_lag_1"]
    c_lag1 = df_ref.loc[i, "Customers_lag_1"]
    df_ref.loc[i, "Sales_per_Customer"] = (
        s_lag1 / c_lag1
        if (not pd.isna(s_lag1) and not pd.isna(c_lag1) and c_lag1 != 0)
        else hist_df_full["Sales_per_Customer"].median()
    )

# Nạp data và model khi mở app

model, spark_session = load_model()
df, features = load_and_preprocess_data()

try:
    # Đọc features.json trực tiếp từ HDFS (qua SparkContext.textFile)
    _features_str = "\n".join(
        spark_session.sparkContext.textFile(FEATURES_JSON_HDFS).collect())
    mllib_features = json.loads(_features_str)
except Exception:
    mllib_features = features

for key in ['bi_report', 'baseline_report', 'df_chart_display']:
    if key not in st.session_state:
        st.session_state[key] = None

tab_home, tab_predict, tab_analytics = st.tabs([
    "TRANG CHỦ", "DỰ BÁO", "THỐNG KÊ"
])