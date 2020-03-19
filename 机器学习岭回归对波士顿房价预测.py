from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor,Ridge
from sklearn.metrics import mean_squared_error

def linear3():
    """
    岭回归对波士顿房价进行预测
    :return:
    """
    # 1）获取数据
    boston = load_boston()
    print("特征数量：\n", boston.data.shape)

    # 2）划分数据集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)

    # 3）标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    #4）预估器
    estimator = Ridge(alpha=0.5, max_iter=10000)
    estimator.fit(x_train, y_train)


    # 5）得出模型
    print("岭回归-权重系数为：\n", estimator.coef_)
    print("岭回归-偏置为：\n", estimator.intercept_)

    # 6）模型评估
    y_predict = estimator.predict(x_test)
    print("预测房价：\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("岭回归-均方误差为：\n", error)

    return None
if __name__ == '__main__':
    linear3()
