# Nhóm 7 – Ứng dụng BigData vào bài toán dự đoán doanh thu của Rossmann

## Phân công
| File | Thành viên | Nhiệm vụ |
|------|-----------|----------|
| `01_data_preparation_and_eda.py` | Tuấn Anh | Load, clean, join, EDA |
| `02_feature_engineering.py` | Hoàng Khang | Tạo features, encoding, dropna |
| `03_mllib_training.py` | Minh Khôi | Train model MLlib, đánh giá |
| `webapp/app.py` + `streaming.py` | Gia Hòa | Load model, Streamlit Web App & Spark Structured Streaming |

## Dataset
Các file được lưu trữ trên HDFS trước khi chạy, ban đầu chỉ cần sale.csv và store.csv, ta chạy lần lượt các file src code 1->3:

| File | Link |
|------|------|
| `sale.csv` + `store.csv` | [Raw data](https://drive.google.com/drive/folders/1yjjBl8fXcmYQTfaf-FVlV2uprl6-d0SG?usp=drive_link) |
| `cleaned_rossmann.csv` | [Sau file 01](https://drive.google.com/file/d/1U2SPjGAxo7mVkUHAaDxQvq3a8kxLMmtZ/view?usp=drive_link) |
| `df_features.csv` + `features.json` | [Sau file 02](https://drive.google.com/drive/folders/1gqdbuG0fkICtWwlQbrTr2WfeetpCyQYw?usp=drive_link) |
| `model/` | [Sau file 03](https://drive.google.com/drive/folders/1Lbw2Do5GbtoJspdapiOLj-DFjxauOBoB?usp=drive_link) |

## Chạy
```bash
pip install pyspark pandas numpy matplotlib seaborn scipy streamlit

# Chạy pipeline lần lượt
python 01_data_preparation_and_eda.py
python 02_feature_engineering.py
python 03_mllib_training.py

# Chạy Web App
streamlit run webapp/app.py

#Chạy Streaming Pipeline

python streaming.py
```
