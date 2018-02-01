import pandas as pd

train=pd.read_csv('train.csv')  #从本地读取训练数据
test=pd.read_csv('test.csv')    #从本地读取测试数据

print(train.info())     #输出训练数据的基本信息
print(test.info())      #输出测试数据的基本信息


selected_features=['Pclass','Sex','Age','Embarked','SibSp','Parch','Fare']

X_train=train[selected_features]
X_test=test[selected_features]
y_train=train['Survived']
print(X_train['Embarked'].value_counts())
print(X_test['Embarked'].value_counts())      #补全Embarked缺失的特征值

X_train['Embarked'].fillna('S',inplace=True)        #用最大的值填充以保证误差最小
X_test['Embarked'].fillna('S',inplace=True)

X_train['Age'].fillna(X_train['Age'].mean(),inplace=True)   #用平均值以保证误差最小
X_test['Age'].fillna(X_test['Age'].mean(),inplace=True)
X_test['Fare'].fillna(X_test['Fare'].mean(),inplace=True)

X_train.info()


from sklearn.feature_extraction import DictVectorizer   #对特征向量化

dict_vec=DictVectorizer(sparse=False)
X_train=dict_vec.fit_transform(X_train.to_dict(orient='record'))
dict_vec.feature_names_
X_test=dict_vec.fit_transform(X_test.to_dict(orient='record'))

