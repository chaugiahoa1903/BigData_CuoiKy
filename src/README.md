# Nhóm 5 – Rossmann Store Sales Forecasting (Big Data)

## Phân công
| File | Thành viên | Nhiệm vụ |
|------|-----------|----------|
| `01_data_preparation_and_eda.py` | Tuấn Anh | Load, clean, join, EDA |
| `02_feature_engineering.py` | Hoàng Khang | Tạo features, encoding, dropna |
| `03_mllib_training.py` | Minh Khôi | Train model MLlib, đánh giá |
| `webapp/app.py` + `webapp/main.py` | Gia Hòa | Load model, Streamlit Web App |
| `utils/` | Cả nhóm | Shared helpers |

## Dataset
Đặt các file sau vào thư mục `data/` trước khi chạy:

| File | Link |
|------|------|
| `sale.csv` + `store.csv` | [Raw data](LINK_DRIVE) |
| `df_clean.csv` | [Sau file 01](LINK_DRIVE) |
| `df_features.csv` + `features.json` | [Sau file 02](LINK_DRIVE) |
| `model/` | [Sau file 03](LINK_DRIVE) |

## Chạy
```bash
pip install pyspark pandas numpy matplotlib seaborn scipy streamlit

# Chạy pipeline lần lượt
python 01_data_preparation_and_eda.py
python 02_feature_engineering.py
python 03_mllib_training.py

# Chạy Web App
streamlit run webapp/app.py
```