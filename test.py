import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, render_template

# 1. Định nghĩa lại class LogisticRegression_pt
class LogisticRegression_pt(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression_pt, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

# Hàm load mô hình và threshold
def load_model(model_path):
    data = torch.load(model_path)
    model = data['model']  # Lấy đối tượng mô hình từ key 'model'
    threshold = data['threshold']  # Lấy giá trị threshold
    model.eval()  # Chuyển sang chế độ đánh giá
    return model, threshold

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

# Tiền xử lý dữ liệu
def preprocess_data(data):
    # Đọc dữ liệu từ file CSV
    data.drop(columns=['Unnamed: 0'], inplace=True)
    if data.shape[1] != 70:
            return render_template('index.html', prediction_text=f"Lỗi: File phải chứa đúng 70 cột đặc trưng (tìm thấy {data.shape[1]} cột).")
    if not all(col in data.columns for col in features):
            return render_template('index.html', prediction_text="Lỗi: Tên cột trong file không khớp với 70 đặc trưng yêu cầu.")
    # print(data.info())
    # print(data.head())
    X_new = torch.tensor(data.values, dtype=torch.float32)
    return data, X_new

# Dự đoán với mô hình
def predict(model, X_tensor, threshold):
    with torch.no_grad():
        outputs = model(X_tensor).numpy()
        predictions = (outputs >= threshold).astype(int)
    return predictions, outputs

# Chạy chương trình
# if __name__ == "__main__":
#      # Đường dẫn tới file model.pth
#     data_path = "unlabel_testcase_data.csv"  # Đường dẫn tới file dữ liệu mới

#     # Load mô hình và threshold
#     try:
#         model, threshold = load_model(model_path)
#         print("Model loaded successfully!")
#         print("Threshold:", threshold)
#         print("Model architecture:", model)
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         exit()

#     # Tiền xử lý dữ liệu
#     try:
#         data, X_tensor = preprocess_data(data_path)
#         print("Data preprocessed successfully!")
#     except Exception as e:
#         print(f"Error preprocessing data: {e}")
#         exit()

#     # Dự đoán
#     # try:
#     #     predictions, outputs = predict(model, X_tensor, threshold)
#     #     data['Predicted Label'] = predictions
#     #     print(data.head)
#     # except Exception as e:
#     #     print(f"Error making predictions: {e}")
#     #     exit()

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', prediction_text=None)

# Route xử lý dự đoán từ file
@app.route('/predict', methods=['POST'])
def predict_by_model():
    try:
        model_path = "D:\\11.NCKH\\model.pth"
        # Kiểm tra file được tải lên
        if 'file' not in request.files:
            return render_template('index.html', prediction_text="Lỗi: Vui lòng tải lên một file.")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction_text="Lỗi: Chưa chọn file.")

        if not file.filename.endswith('.csv'):
            return render_template('index.html', prediction_text="Lỗi: Vui lòng tải lên file .csv.")

        # Tiền xử lý dữ liệu
        try:
            data, X_tensor = preprocess_data(pd.read_csv(file))
            print("Data preprocessed successfully!")
        except Exception as e:
            print(f"Error preprocessing data: {e}")
            exit()

        # Load mô hình và threshold
        try:
            model, threshold = load_model(model_path)
            print("Model loaded successfully!")
            print("Threshold:", threshold)
            print("Model architecture:", model)
        except Exception as e:
            print(f"Error loading model: {e}")
            exit()

         # Dự đoán
        try:
            predictions, outputs = predict(model, X_tensor, threshold)
            data['Predicted Label'] = predictions
            print(data.head)
            print(type(data))
            prediction_text = data.to_html(classes='table table-striped table-bordered', index=False)
        except Exception as e:
            print(f"Error making predictions: {e}")
            exit()
        return render_template('index.html', prediction_text=prediction_text)
    except Exception as e:
        return render_template('index.html', prediction_text=f"Lỗi: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)