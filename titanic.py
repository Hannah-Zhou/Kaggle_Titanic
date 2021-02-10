#!/usr/bin/env python
# coding: utf-8

# # 泰坦尼克之灾实验报告

# 导包

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import re
import warnings
warnings.simplefilter('ignore')


# # 一、数据预处理：缺失值处理+字符串转数字

# In[2]:


train_set = pd.read_csv('./train.csv')
test_set = pd.read_csv('./test.csv')
train_set.head()


# In[3]:


train_set.info()
test_set.info()


# In[4]:


train_set.describe()


# In[5]:


# 数据集合并
train_test = pd.concat([train_set, test_set])
train_test.info()


# In[6]:


# 查看每列是否有缺失情况
train_test.isnull().any()


# In[7]:


# 缺失值处理
train_test['Age'].isnull().value_counts()


# In[8]:


# 处理age的缺失,用中位数填充,并修改原数据集
train_test['Age'].fillna(train_test['Age'].median(),inplace = True)
train_test['Age']


# In[9]:


train_test['Embarked'].unique()


# In[10]:


train_test['Embarked'].value_counts()


# In[11]:


# 处理embarked的缺失，用众数填充,并保存修改
train_test['Embarked'].fillna('S',inplace = True)


# In[12]:


train_test['Embarked'].value_counts()


# In[13]:


# 字符串转数字
train_test['Sex'] = train_test['Sex'].map({'female':0, 'male':1})
train_test['Embarked'] = train_test['Embarked'].map({'S':0, 'C':1, 'Q':2})


# In[14]:


train_test.info()


# In[15]:


# 缺失值多，去除cabin
train_test.drop('Cabin',axis=1,inplace = True)


# In[16]:


train_test.head()


# In[17]:


# 用皮尔逊系数计算各特征相关性，并用热力图可视化
train_test_corr = train_test.corr()
plt.figure(figsize = (8,8))
ax = plt.subplot()
ax.set_xticklabels(train_test_corr.columns)
ax.set_yticklabels(train_test_corr.index)
plt.imshow(train_test_corr)  
plt.colorbar()


# In[18]:


train_test_corr


# ### 用皮尔逊系数计算各特征相关性，并用热力图可视化

# In[19]:


plt.subplots(figsize=(10,10))
sns.heatmap(train_test_corr,vmin = -1, annot = True, square = True)


# In[20]:


# 处理fare，根据热力图和经验，fare与pclass和embarked有关
train_test.groupby(['Pclass', 'Embarked'])['Fare'].mean()


# In[21]:


# 查看缺失值
train_test[train_test['Fare'].isnull()]


# In[22]:


# 用对应平均值填充fare缺失值
train_test['Fare'].fillna(14.435422,inplace = True)


# In[23]:


train_test.info()


# In[24]:


# 数据预处理完成，保留有用特征，划分数据集
# features = ['Survived'，'Pclass', 'Sex', 'Age', 'SibSp','Parch',  'Fare', 'Embarked']
train_data = train_test[:891]
test_data = train_test[891:]


# In[25]:


train_data.shape


# In[26]:


train_data.columns


# In[27]:


train_data['Survived']


# In[28]:


train_data[['SibSp','Parch']].T


# features = ['Survived'，'Pclass', 'Sex', 'Age', 'SibSp','Parch',  'Fare', 'Embarked']

# # 二、乘客属性分布

# In[29]:


#乘客属性分布 绘图
# 设置属性
plt.rcParams['font.sans-serif'] = ['SimHei'] # 正常显示中文标签
plt.rcParams['font.family']='sans-serif' 
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号
fig = plt.figure(figsize=(20,10))
fig.set(alpha=0.2) # 设定图表颜色alpha参数
# 获救情况 survived 柱状图
plt.subplot2grid((2,4),(0,0)) 
train_data['Survived'].value_counts().plot(kind='bar')
plt.title("获救情况 (1为获救)")
plt.ylabel("人数") 
# 等级分布 pclass 柱状图
plt.subplot2grid((2,4),(0,1))
train_data.Pclass.value_counts().plot(kind="bar") 
plt.ylabel("人数")
plt.title("乘客等级分布")
# 性别分布 sex 柱状图
plt.subplot2grid((2,4),(0,2))
train_data.Sex.value_counts().plot(kind="bar") 
plt.ylabel("人数")
plt.title("乘客性别分布")
# 年龄分布 age 散点图
plt.subplot2grid((2,4),(0,3))
x = np.linspace(0,891,891)
plt.scatter(x,train_data['Age'])
plt.ylabel("年龄")
plt.title("乘客年龄分布")
# # 年龄 age 散点图
# plt.subplot2grid((2,4),(0,4))
# plt.scatter(train_data['Survived'], train_data['Age']) 
# plt.ylabel("年龄") 
# plt.grid(b=True, which='major', axis='y')
# plt.title("按年龄看获救分布 (1为获救)")
# 各等级的乘客年龄分布 密度图
plt.subplot2grid((2,4),(1,0), colspan=2)
train_data.Age[train_data.Pclass == 1].plot(kind='kde')  
train_data.Age[train_data.Pclass == 2].plot(kind='kde')
train_data.Age[train_data.Pclass == 3].plot(kind='kde')
plt.xlabel("年龄")
plt.ylabel("密度")
plt.title("各等级的乘客年龄分布")
plt.legend(('头等舱', '2等舱','3等舱'),loc='best')
# 船票分布 fare 散点图
plt.subplot2grid((2,4),(1,2))
plt.scatter(x,train_data['Fare'])
plt.xlabel("船票费用")
plt.title("船票费用分布")
# 各登船口岸上船人数 柱状图
plt.subplot2grid((2,4),(1,3))
train_data.Embarked.value_counts().plot(kind='bar')
plt.title("各登船口岸上船人数")
plt.ylabel("人数")
plt.show()


# # 三、按特征分别观察与获救的关系：特征组合+创建新特征

# In[30]:


# 乘客等级与获救情况 
Survived_0 = train_data.Pclass[train_data.Survived == 0].value_counts() # 0 未获救
Survived_1 = train_data.Pclass[train_data.Survived == 1].value_counts() # 1 获救
df = pd.DataFrame({'获救':Survived_1,'未获救':Survived_0})
df.plot(kind = 'bar', stacked = True)
plt.title('各乘客等级的获救情况')
plt.xlabel('乘客等级')
plt.ylabel('人数')
plt.show()


# In[31]:


# 按性别看获救情况 'female':0, 'male':1
Survived_m = train_data.Survived[train_data.Sex == 1].value_counts() # m 男性
Survived_f = train_data.Survived[train_data.Sex == 0].value_counts()# f 女性
df = pd.DataFrame({'男性':Survived_m,'女性':Survived_f})
df.plot(kind = 'bar', stacked = True)
plt.title('按性别看获救情况')
plt.xlabel('是否获救')
plt.ylabel('人数')
plt.show()


# In[32]:


#联合分析，等级12与3差异大，男女差异大，根据舱等级（12为中高 3为低）和性别的获救情况 
fig = plt.figure(figsize=(20,5))
plt.title('联合分析舱等级和性别的获救情况')
# 女性 0 高级舱 
ax1 = fig.add_subplot(141) 
train_data.Survived[train_data.Sex == 0][train_data.Pclass != 3].value_counts().plot(kind = 'bar', label = 'female medium high class', color = 'red')
ax1.set_xticklabels(['获救','未获救'], rotation = 0) 
ax1.legend(['女性/中高级舱'], loc = 'best')

ax2 = fig.add_subplot(142, sharey = ax1) 
train_data.Survived[train_data.Sex == 0][train_data.Pclass == 3].value_counts().plot(kind = 'bar', label = 'female low class', color = 'pink')
ax2.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"女性/低级舱"], loc='best')

ax3 = fig.add_subplot(143, sharey = ax1)
train_data.Survived[train_data.Sex == 1][train_data.Pclass != 3].value_counts().plot(kind = 'bar', label = 'male medium high class', color = 'blue')
ax3.set_xticklabels(['未获救','获救'], rotation = 0)
plt.legend(['男性/中高级舱'], loc = 'best')

ax4 = fig.add_subplot(144, sharey = ax1)
train_data.Survived[train_data.Sex == 1][train_data.Pclass == 3].value_counts().plot(kind = 'bar', label = 'male low class', color = 'cyan')
ax4.set_xticklabels(['未获救','获救'], rotation = 0)
plt.legend(['男性/低级舱'], loc = 'best')
plt.show()


# In[33]:


# 各登陆港口乘客的获救情况
fig = plt.figure()
Survived_0 = train_data.Embarked[train_data.Survived == 0].value_counts()
Survived_1 = train_data.Embarked[train_data.Survived == 1].value_counts()
df = pd.DataFrame({'获救':Survived_1,'未获救':Survived_0})
df.plot(kind = 'bar', stacked = True)
plt.title('各登陆港口乘客的获救情况')
plt.xlabel('登陆港口')
plt.ylabel('人数')
plt.show()


# In[34]:


# 堂兄妹人数与获救 分组分析
g = train_data.groupby(['SibSp','Survived']) 
df = pd.DataFrame(g.count()['PassengerId']).T
df


# In[35]:


# 父母人数与获救 
g = train_data.groupby(['Parch','Survived']) 
df = pd.DataFrame(g.count()['PassengerId']).T
df


# ### 创建新特征SibSp_Parch，联合分析亲人人数与获救关系 

# In[36]:


# 创建新特征SibSp_Parch，联合分析亲人人数与获救关系 
train_test['SibSp_Parch'] = train_test['SibSp'] + train_test['Parch']
g = train_data.groupby(['SibSp_Parch','Survived']) 
df = pd.DataFrame(g.count()['PassengerId']).T
df


# ### 创建名称（称呼可能体现地位）新特征 NameLength, Title，分类处理

# In[178]:


train_test['NameLength'] = train_test['Name'].map(lambda name: len(name))
train_test['NameLength'].head()


# In[185]:


# 正则匹配name,提取称呼，如 'Carlsson, Mr. Frans Olof'
train_test['Title'] = train_test['Name'].map(lambda name: re.search(r', (.*?)\. .*', name).group(1))
train_test['Title'].unique()


# In[186]:


# col表示上校, rev表示牧师, Mlle表示法国小姐, Major陆军少校, Sir爵士, Capt上尉, Countess女伯爵, Jonkheer无名贵族, Don阁下,尊称,Mme夫人 Dona葡萄牙语的夫人小姐
# 把相同的意思的title归成一类.
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9,"Dona":10,"Lady": 10, "the Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
train_test['Title'] = train_test['Title'].map(title_mapping)
train_test['Title'].unique()


# In[187]:


train_test.head()


# In[188]:


train_test.info()


# In[189]:


train_test[train_test['Title'].isnull()]


# In[ ]:





# In[ ]:





# # 四、 创建模型，分别使用新老特征预测

# In[191]:


features_old = ['Survived','Pclass', 'Sex', 'Age', 'SibSp','Parch',  'Fare', 'Embarked']
features_new = ['Survived','Pclass', 'Sex', 'Age','Fare','Embarked', 'SibSp_Parch','NameLength', 'Title']
train_data = train_test[:891]
test_data = train_test[891:]


# ### 逻辑斯蒂回归模型预测

# In[42]:


# 逻辑斯蒂回归模型
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[43]:


# 老特征
data_old = train_data[features_old].drop(['Survived'],axis = 1)
target_old = train_data[features_old]['Survived']
X_train_old, X_test_old, y_train_old, y_test_old = train_test_split(data_old,target_old,test_size=0.3)


# In[44]:


logisitc = LogisticRegression(random_state=0)
logisitc.fit(X_train_old, y_train_old)


# In[45]:


y_predict = logisitc.predict(X_test_old)


# In[46]:


score = logisitc.score(X_test_old, y_test_old)
score


# In[47]:


# 新特征
data_new = train_data[features_new].drop(['Survived'],axis = 1)
target_new = train_data[features_new]['Survived']
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(data_new,target_new,test_size=0.3)


# In[48]:


logisitc = LogisticRegression()
logisitc.fit(X_train_new, y_train_new)
logisitc.score(X_test_new, y_test_new)


# #### 交叉验证优化 优点：
# 1：交叉验证用于评估模型的预测性能，尤其是训练好的模型在新数据上的表现，可以在一定程度上减小过拟合。
# 2：还可以从有限的数据中获取尽可能多的有效信息。
# 原始采用的train_test_split方法，数据划分具有偶然性；交叉验证通过多次划分，大大降低了这种由一次随机划分带来的偶然性，同时通过多次划分，多次训练，模型也能遇到各种各样的数据，从而提高其泛化能力；
# 与原始的train_test_split相比，对数据的使用效率更高。train_test_split，默认训练集、测试集比例为3:1，而对交叉验证来说，如果是5折交叉验证，训练集比测试集为4:1；10折交叉验证训练集比测试集为9:1。数据量越大，模型准确率越高！

# In[49]:


from sklearn.model_selection import cross_val_score


# In[50]:


#老特征
logisitc = LogisticRegression()
score = cross_val_score(logisitc,data_old, target_old, cv=5).mean()
score


# In[51]:


# 老特征
logisitc = LogisticRegression()
score = cross_val_score(logisitc,data_old, target_old, cv=10).mean()
score


# In[52]:


# 新特征
logisitc = LogisticRegression()
score = cross_val_score(logisitc,data_new, target_new, cv=5).mean()
score


# In[53]:


# 新特征
data = train_data[features_new].drop(['Survived'],axis = 1)
target = train_data[features_new]['Survived']
logisitc = LogisticRegression()
score = cross_val_score(logisitc,data_new, target_new, cv=10).mean()
score


# ### 使用随机森林预测

# In[54]:


from sklearn.ensemble import RandomForestClassifier


# In[55]:


# 老特征
rfc = RandomForestClassifier(random_state=0)
rfc.fit(X_train_old, y_train_old)
rfc.score(X_test_old, y_test_old)


# In[56]:


# 新特征
rfc = RandomForestClassifier(random_state=0)
rfc.fit(X_train_new, y_train_new)
rfc.score(X_test_new, y_test_new)


# #### 交叉验证优化

# In[57]:


# 老特征
rfc = RandomForestClassifier(random_state=0)
score = cross_val_score(rfc,data, target, cv=5).mean()
score


# In[58]:


# 新特征
rfc = RandomForestClassifier(random_state=0)
score = cross_val_score(rfc,data, target, cv=10).mean()
score


# 初步小结：新特征优化模型score更高；随机森林分类score高于逻辑斯蒂模型；交叉验证优化计算得出score比train_test_split结果更稳定，train_test_split只进行了一次划分，数据结果具有偶然性，如果在某次划分中，训练集里全是容易学习的数据，测试集里全是复杂的数据，这样就会导致最终的结果不尽如意，所以交叉验证的评价结果更加有参考意义。一般k为5或者10

# 网格搜索定义：
# 网格搜索法是指定参数值的一种穷举搜索方法，通过将估计函数的参数通过交叉验证进行优化来得到最优的学习算法。
# 
# 步骤：
# · 将各个参数可能的取值进行排列组合，列出所有可能的组合结果生成“网格”。
# 
# · 然后将各组合用于SVM训练，使用交叉验证对表现进行评估。
# 
# · 在拟合函数尝试了所有的参数组合后，返回一个合适的分类器，自动调整至最佳（性能度量）参数组合，可以通过clf.best_params_获得参数值
# 
# 使用交叉验证对模型评估：
# 如果使用K折交叉验证。将原始数据划分为k份，k-1份作为训练集，1份作为测试集。轮流进行k次。
# 
# 性能度量：
# 可以选择accuracy, f1-score, f-beta, precise, recall等
# 'criterion':("gini","entropy")

# ### 随机森林调参优化， 用网格搜索调整参数

# In[59]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


parameters = {'criterion':("gini","entropy")
,"max_depth":[*range(1,10)]
,"min_samples_split":[*range(1,10)]
,'min_samples_leaf':[*range(1,10)]
}
rfc = RandomForestClassifier(random_state=1)
GS = GridSearchCV(rfc, parameters, cv=10)
GS.fit(X_train_new, y_train_new)
GS.best_params_
GS.best_score_


# In[61]:


parameters = {
"min_samples_split":[*range(1,10)]
,'min_samples_leaf':[*range(1,10)]
}
rfc = RandomForestClassifier(random_state=2)
GS = GridSearchCV(rfc, parameters, cv=10)
GS.fit(X_train_new, y_train_new)
GS.best_params_
GS.best_score_


# In[62]:


GS.best_params_
GS.best_score_


# In[63]:


GS.best_params_


# In[68]:


pd.DataFrame(GS.cv_results_).T


# In[69]:


parameters = {
"min_samples_split":[*range(1,10)]
,'min_samples_leaf':[*range(1,10)]
}
rfc = RandomForestClassifier(random_state=2)
GS = GridSearchCV(rfc, parameters, cv=10)
GS.fit(data_new, target_new)
GS.best_params_
GS.best_score_


# In[70]:


GS.best_params_


# In[71]:


pd.DataFrame(GS.cv_results_).T


# In[72]:


parameters = {
"min_samples_split":[*range(1,10)]
,'min_samples_leaf':[*range(1,10)]
}
rfc = RandomForestClassifier(random_state=2)
GS = GridSearchCV(rfc, parameters, cv=10)
GS.fit(data_old, target_old)
GS.best_params_
GS.best_score_


# In[73]:


GS.best_params_


# In[139]:


rfc = RandomForestClassifier(n_estimators=100, min_samples_split=7, min_samples_leaf=2)
rfc.fit(data_new, target_new)
score = cross_val_score(rfc,data_new, target_new, cv=10).mean()
score


# ### 筛选主要特征预测

# In[140]:


importances = rfc.feature_importances_
importances


# #### 绘制柱状图比较重要性

# In[141]:


plt.figure(figsize=(10,8))
plt.bar(np.arange(0,8), importances)
_ = plt.xticks(np.arange(0,8), features_new[1:])


# #### 小结：通过对随机森林网格优化调参，采取n_estimators=100, min_samples_split=7, min_samples_leaf=2的随机森林，
# #### 选取特征为['Pclass', 'Sex', 'Age','Fare','NameLength', 'Title','SibSp_Parch','Embarked']，交叉验证后得到的模型得分最高。
# #### 然后对此模型的选取特征的重要性进行评估，得到重要性值并绘制柱状图，发现'SibSp_Parch','Embarked'特征的重要性低，所以采取去掉这两个对模型再次训练。
# #### 结果发现，虽然重要性低，仍然对模型得分产生影响，故仍保留这两个特征。该模型得分可到84左右。

# In[147]:


features=['Pclass', 'Sex', 'Age','Fare','NameLength', 'Title']
rfc = RandomForestClassifier(n_estimators=100, min_samples_split=7, min_samples_leaf=2)
rfc.fit(data_new[features], target_new)
score = cross_val_score(rfc,data_new[features], target_new, cv=10).mean()
score


# In[ ]:





# In[148]:


features=['Pclass', 'Sex', 'Age','Fare','NameLength', 'Title','Embarked']
rfc = RandomForestClassifier(n_estimators=100, min_samples_split=7, min_samples_leaf=2)
rfc.fit(data_new[features], target_new)
score = cross_val_score(rfc,data_new[features], target_new, cv=10).mean()
score


# In[149]:


features=['Pclass', 'Sex', 'Age','Fare','NameLength', 'Title','SibSp_Parch']
rfc = RandomForestClassifier(n_estimators=100, min_samples_split=7, min_samples_leaf=2)
rfc.fit(data_new[features], target_new)
score = cross_val_score(rfc,data_new[features], target_new, cv=10).mean()
score


# In[151]:


features=['Pclass', 'Sex', 'Age','Fare','NameLength', 'Title','SibSp_Parch','Embarked']
rfc = RandomForestClassifier(n_estimators=100, min_samples_split=7, min_samples_leaf=2)
rfc.fit(data_new[features], target_new)
score = cross_val_score(rfc,data_new[features], target_new, cv=10).mean()
score


# In[192]:


test_data[features].head()


# In[193]:


data_test = test_data[features]
data_test.info()


# In[ ]:





# ### 用模型预测测试集，输出结果

# In[195]:


target_test = rfc.predict(data_test)


# In[203]:


result = pd.DataFrame({"Survived":target_test},index=test_data["PassengerId"])


# In[204]:


result.head()


# In[205]:


result.to_csv('result.csv', header=True, index=True)

