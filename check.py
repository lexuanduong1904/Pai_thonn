import pandas as pd

data = pd.read_csv('unlabel_testcase_data.csv')
data.drop(columns=['Unnamed: 0'], inplace=True)
print(data.shape)