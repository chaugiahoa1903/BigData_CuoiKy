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

