import pandas as pd 
#获取数据
data = pd.read_csv("./jiqixuexi1/train.csv")
s = data.head()
print(s)