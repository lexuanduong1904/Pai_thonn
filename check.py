import torch
import torch.nn as nn

# Định nghĩa class LogisticRegression_pt
class LogisticRegression_pt(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression_pt, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # Giả định đơn giản
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

# Kiểm tra nội dung file .pth
data = torch.load("model.pth", weights_only=True)
print(type(data))
print(data)