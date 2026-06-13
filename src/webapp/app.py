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


