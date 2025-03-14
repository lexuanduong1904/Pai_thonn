from flask import Flask, request, render_template
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Định nghĩa lớp mô hình PyTorch
class LogisticRegression_pt(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression_pt, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

app = Flask(__name__)

# Tải mô hình
try:
    data = joblib.load('model_lr.pkl')
    model = data['model']
    threshold = data['threshold']
    model.eval()
except FileNotFoundError:
    print("File 'model_lr.pkl' không tồn tại. Vui lòng đặt file trong thư mục dự án.")
    exit(1)
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
    exit(1)

# Danh sách 70 đặc trưng
features = [
    'Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
    'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min',
    'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min',
    'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot',
    'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Tot',
    'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
    'Fwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s',
    'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var',
    'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt',
    'URG Flag Cnt', 'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg',
    'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Subflow Fwd Pkts', 'Subflow Fwd Byts',
    'Subflow Bwd Pkts', 'Subflow Bwd Byts', 'Init Fwd Win Byts', 'Init Bwd Win Byts',
    'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max',
    'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
]

# Chuẩn hóa (scaler chưa fit, cần dữ liệu gốc để fit)
scaler = StandardScaler()

# Route trang chủ
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', prediction_text=None)

# Route xử lý dự đoán từ file
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Kiểm tra file được tải lên
        if 'file' not in request.files:
            return render_template('index.html', prediction_text="Lỗi: Vui lòng tải lên một file.")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction_text="Lỗi: Chưa chọn file.")
        
        if not file.filename.endswith('.csv'):
            return render_template('index.html', prediction_text="Lỗi: Vui lòng tải lên file .csv.")

        # Đọc file CSV có tiêu đề
        input_data = pd.read_csv(file)
        
        # Loại bỏ cột đầu tiên (không tên) và giữ 70 cột đặc trưng
        if input_data.columns[0] not in features:  # Nếu cột đầu tiên không phải đặc trưng
            input_data = input_data.iloc[:, 1:]  # Loại bỏ cột đầu tiên
        
        # Kiểm tra số cột
        if input_data.shape[1] != 70:
            return render_template('index.html', prediction_text=f"Lỗi: File phải chứa đúng 70 cột đặc trưng (tìm thấy {input_data.shape[1]} cột).")
        
        # Kiểm tra tên cột
        if not all(col in input_data.columns for col in features):
            return render_template('index.html', prediction_text="Lỗi: Tên cột trong file không khớp với 70 đặc trưng yêu cầu.")
        
        # Chọn đúng 70 cột đặc trưng theo thứ tự
        input_data = input_data[features].values  # Chuyển thành numpy array
        
        # Chuẩn hóa dữ liệu
        input_data_scaled = scaler.transform(input_data)
        
        # Chuyển thành tensor
        input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)
        
        # Dự đoán
        model.eval()
        with torch.no_grad():
            probs = model(input_tensor).numpy().flatten()  # Xác suất cho từng mẫu
            predictions = (probs > threshold).astype(int)  # Phân loại nhị phân
            results = ["Tấn công" if pred == 1 else "Bình thường" for pred in predictions]
            
            # Tạo kết quả chi tiết
            result_text = "<h4>Kết quả dự đoán:</h4><ul>"
            for i, (prob, result) in enumerate(zip(probs, results)):
                result_text += f"<li>Mẫu {i+1}: {result} (Xác suất: {prob:.4f})</li>"
            result_text += "</ul>"

        return render_template('index.html', prediction_text=result_text)
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Lỗi: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)