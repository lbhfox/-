from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
def knn_iris():
    #knn算法对鸢尾花进行分类

    #数据获取
    iris = load_iris()
    #数据集划分
    x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,random_state=6)
    #特征工程：标准化 
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)
    #KNN算法预估器
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train,y_train)
    #模型评估
    y_predict = estimator.predict(x_test)
    print("y_predict:\n",y_predict)
    print("直接比对真实值禾预测值：\n",y_test==y_predict)
    score = estimator.score(x_test,y_test)
    print("准确率:\n",score)
    return None 

if __name__ == "__main__":
    knn_iris()