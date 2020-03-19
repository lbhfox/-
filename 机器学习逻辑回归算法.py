from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
def cancer():
    # 1、读取数据
    path = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
    column_name = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                   'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                   'Normal Nucleoli', 'Mitoses', 'Class']

    data = pd.read_csv(path, names=column_name)
    # 2、缺失值处理
    # 1）替换-》np.nan
    data = data.replace(to_replace="?", value=np.nan)
    # 2）删除缺失样本
    data.dropna(inplace=True)
    data.isnull().any()  # 检验,不存在缺失值
    # 3、划分数据集
    # 筛选特征值和目标值
    x = data.iloc[:, 1:-1]
    y = data["Class"]
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    # 4、标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 5、预估器流程
    estimator = LogisticRegression()
    estimator.fit(x_train, y_train)
    # 5）得出模型
    print("逻辑回归-权重系数为：\n", estimator.coef_)
    print("逻辑回归-偏置为：\n", estimator.intercept_)
    # 6、模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接比对真实值和预测值:\n", y_test == y_predict)

    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：\n", score)
    # 查看精确率、召回率、F1-score
    report = classification_report(y_test, y_predict, labels=[2, 4], target_names=["良性", "恶性"])
    print(report)
    # y_true：每个样本的真实类别，必须为0(反例),1(正例)标记
    # 将y_test 转换成 0 1
    y_true = np.where(y_test > 3, 1, 0)
    score = roc_auc_score(y_true, y_predict)
    print(score)
if __name__ == '__main__':
    cancer()
