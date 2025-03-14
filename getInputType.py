import joblib
import torch
import torch.nn as nn
# Định nghĩa lớp trước khi tải
class LogisticRegression_pt(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression_pt, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Tải file
data = joblib.load('model_lr.pkl')

# Truy cập mô hình và ngưỡng
model = data['model']
threshold = data['threshold']

print("Model:", model)
print(type(model))
print("Threshold:", threshold)