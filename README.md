# Solar Power Analysis Project

Dự án này tập trung vào việc phân tích dữ liệu phát điện mặt trời và dữ liệu cảm biến thời tiết từ hai nhà máy năng lượng mặt trời (Plant 1 và Plant 2). Mục tiêu chính là làm sạch dữ liệu, phân tích các yếu tố ảnh hưởng đến sản lượng điện và xây dựng mô hình dự báo công suất AC (AC Power) dựa trên các điều kiện thời tiết và thời gian.

## 📌 Tính năng chính

*   **Xử lý & Làm sạch dữ liệu**: Tự động tải và chuẩn hóa định dạng thời gian cho dữ liệu phát điện và thời tiết từ nhiều nguồn khác nhau.
*   **Kỹ thuật đặc trưng (Feature Engineering)**: 
    *   Trích xuất các đặc trưng thời gian (giờ, ngày, tháng, tuần).
    *   Chuyển đổi giờ sang dạng chu kỳ (Sin/Cos) để mô hình học tập tốt hơn.
    *   Xác định khung giờ có ánh sáng mặt trời (Is Daylight).
*   **Tổng hợp dữ liệu**: Gộp dữ liệu từ các bộ biến tần (inverters) để tính tổng công suất toàn nhà máy.
*   **Trực quan hóa**:
    *   So sánh sản lượng điện và cường độ bức xạ (Irradiation) hàng ngày giữa các nhà máy.
    *   Biểu đồ nhiệt (Heatmap) thể hiện tương quan giữa các biến.
    *   Biểu đồ so sánh kết quả dự báo và thực tế.
    *   Đánh giá mức độ quan trọng của các đặc trưng (Feature Importance).
*   **Học máy (Machine Learning)**: Sử dụng mô hình `RandomForestRegressor` để dự báo công suất điện với quy trình Pipeline tích hợp tiền xử lý dữ liệu.

## 📂 Cấu trúc thư mục

*   `datasets/`: Chứa các file dữ liệu đầu vào (`Plant_X_Generation_Data.csv`, `Plant_X_Weather_Sensor_Data.csv`).
*   `outputs/`: Chứa các kết quả phân tích bao gồm biểu đồ (PNG), báo cáo (Markdown), và các file CSV đã qua xử lý.
*   `solar_power_analysis.py`: File thực thi chính của dự án.

## 🛠 Yêu cầu hệ thống

Dự án yêu cầu Python 3.8+ và các thư viện sau:
*   `pandas`
*   `numpy`
*   `matplotlib`
*   `scikit-learn`

Bạn có thể cài đặt nhanh qua pip:
```bash
pip install pandas numpy matplotlib scikit-learn
```

## 🚀 Cách sử dụng

1. Đảm bảo dữ liệu đã được đặt đúng trong thư mục `datasets/` hoặc cùng cấp với file script.
2. Chạy script chính:
   ```bash
   python solar_power_analysis.py
   ```
3. Sau khi hoàn tất, kiểm tra thư mục `outputs/` để xem báo cáo `analysis_report.md` và các biểu đồ phân tích.

## 📊 Kết quả mô hình

Mô hình được đánh giá dựa trên các chỉ số:
*   **MAE** (Mean Absolute Error)
*   **RMSE** (Root Mean Squared Error)
*   **R2 Score** (Hệ số xác định)

Chi tiết kết quả cho từng nhà máy được lưu tại `outputs/metrics.json`.

---
*Dự án được thực hiện bởi Nhóm 13 - KPDL.*