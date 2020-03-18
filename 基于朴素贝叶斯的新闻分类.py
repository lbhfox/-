#用朴素贝叶斯算法对新闻进行分类
#1获取数据
#划分数据集
#特征工程：文本特征抽取
#朴素贝叶斯预估器
#模型评估
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def nb_news():
    #获取数据
    news = fetch_20newsgroups(subset="all")
    #划分数据集
   x_train,x_test,y_train,y_test =  train_test_split(news.data.news.target)
    #特征工程
    transfer = TfidfVectorizer()
    x_train = transger.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)
    #朴素贝叶斯算法预估器流程
    estimator = MultinomialNB()
    estimator.fit(x_train,y_train)
    #模型评估
     y_predict = estimator.predict(x_test)
    print("y_predict:\n",y_predict)
    print("直接比对真实值禾预测值：\n",y_test==y_predict)
    score = estimator.score(x_test,y_test)
    print("准确率:\n",score)
    print("jieguo",estimator.best_score_)
    print("参数",estimator.best_params_)

if __name__ == "__main__":
   nb_news()