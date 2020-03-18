from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
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
    #加入网格搜索与交叉验证
    #参数准备
    param_dict = {"n_neighbors":[1,3,5,7,9,11]}
    estimator = GridSearchCV(estimator,param_grid=param_dict,cv=10)
    estimator.fit(x_train,y_train)
    #模型评估
    y_predict = estimator.predict(x_test)
    print("y_predict:\n",y_predict)
    print("直接比对真实值禾预测值：\n",y_test==y_predict)
    score = estimator.score(x_test,y_test)
    print("准确率:\n",score)
    print("jieguo",estimator.best_score_)
    print("参数",estimator.best_params_)
    return None 
def decision_iris():
    #用决策树对鸢尾花进行分类
    #获取数据集
    iris = load_iris()

    #划分数据集
   x_train,x_test,y_train,y_test =  train_test_split(iris.data,iris.target,random_state=22)
    #决策树预估器
   estimator =  DecisionTreeClassifier(criterion="entropy")
   estimator.fit(x_train,y_train)
    #模型评估
     #模型评估
    y_predict = estimator.predict(x_test)
    print("y_predict:\n",y_predict)
    print("直接比对真实值禾预测值：\n",y_test==y_predict)
    score = estimator.score(x_test,y_test)
    print("准确率:\n",score)
    print("jieguo",estimator.best_score_)
    print("参数",estimator.best_params_)
    return None 

if __name__ == "__main__":
    decision_iris()