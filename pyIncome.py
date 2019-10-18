#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
get_ipython().run_line_magic('pylab', 'inline')
import seaborn as sns
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from xgboost import XGBClassifier
from wordcloud import WordCloud, STOPWORDS,ImageColorGenerator


# In[2]:


adult=pd.read_csv('adult.csv')
adult.head()


# In[3]:


adult.shape


# # DATASET

# | This data was extracted from the census bureau database found at http://www.census.gov/ftp/pub/DES/www/welcome.html <br>Donor: Ronny Kohavi and Barry Becker, Data Mining and Visualization Silicon Graphics<br>.
# | e-mail: ronnyk@sgi.com for questions<br>.
# | Split into train-test using MLC++ GenCVFiles (2/3, 1/3 random)<br>.
# | 48842 instances, mix of continuous and discrete    (train=32561, test=16281)<br>
# | 45222 if instances with unknown values are removed (train=30162, test=15060)<br>
# | Duplicate or conflicting instances : 6<br>
# | Class probabilities for adult.all file<br>
# | Probability for the label '>50K'  : 23.93% / 24.78% (without unknowns)<br>
# | Probability for the label '<=50K' : 76.07% / 75.22% (without unknowns)<br>
# |<br>
# | Extraction was done by Barry Becker from the 1994 Census database.  A set of<br>
# | reasonably clean records was extracted using the following conditions:<br>
# |   ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
# |<br>
# | Prediction task is to determine whether a person makes over 50K a year<br>.
# |
# |First cited in:
# | @inproceedings{kohavi-nbtree,
# |    author={Ron Kohavi},
# |    title={Scaling Up the Accuracy of Naive-Bayes Classifiers: a
# |           Decision-Tree Hybrid},
# |    booktitle={Proceedings of the Second International Conference on
# |               Knowledge Discovery and Data Mining},
# |    year = 1996,
# |    pages={to appear}}<br>
# |<br>
# |Error Accuracy reported as follows, after removal of unknowns from train/test sets):<br><br>
# |   C4.5       : 84.46+-0.30<br>
# |   Naive-Bayes: 83.88+-0.30<br>
# |   NBTree     : 85.90+-0.28<br>

# # OBJECTIF 

# The objectif of this exercise is to build a model capable of predicting if an individual is earning over 50K that is  better than the previous ones.<br>
# We will try to beat the accuracy of the previous created model (about 84%). <br>
# However, accuracy could be very much misleading so therefore our first objectif would remain to build a reliable
# model bearing in mind that a benchmark has already been established by Naive-Bayes model (83.88+-0.30)<br>
# *The other pupose of this tutorial is to show the importance of Data preprocessing and showing how to deal with imbalanced classes of outcome variables*.

# # EXPLORATION

# Full exploration performed on perform is usually starts with having a great overview of the dataset features <br>
# (by calling the __.info(
# )__ method on the dataset) then having a description of the dataset you are about the work on.<br>
# However,the steps can be skipped by the using the **Pandas_Profiling** module that provides a much deeper description of dasets

# In[4]:


ProfileReport(adult)


# # To Flag

# - Missing data seems represented by *'?'* <br>
# - **educational_num** seems to be an encoding of the **education** variable (categorical version will be dropped)<br>
# - The **income** is unbalanced (That could be a problem for the modelling let's see)<br>
# - Few duplicates  are counted in the file
# 

# In[ ]:





# # QUICK VIZ

# In[5]:


sns.set(rc={'figure.figsize':(9.7,7.27)})
mask = np.zeros_like(adult.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

sns.set_style('white')
cmap = sns.diverging_palette(220, 10, as_cmap=True)
plt.figure(figsize=(12,7))
sns.heatmap(adult.corr(), mask=mask, cmap=cmap, vmax=.8, center=0,square=True, linewidths=.5, 
            cbar_kws={"shrink": .4}, annot=True);


# In[6]:


sns.set_style('whitegrid')
sns.lmplot(x='age', y='fnlwgt', data=adult, hue='gender');


# In[7]:


sns.set_style('whitegrid')
plt.figure(figsize=(8,5))
plt.hist('age', data=adult, color='red');
plt.title('Age Distribution', fontsize=15)


# In[8]:


plt.figure(figsize=(8,5))
plt.hist(np.log(adult.age))


# In[9]:


plt.hist(np.cbrt(adult.age))


# In[10]:


sns.set_style('whitegrid')
adult.groupby(['gender','race'])['fnlwgt'].mean().plot(kind='barh', figsize=(7,5),
                                                      width=.6, title='FNLWGT per Race and ');


# # PREPARATION OF FEATURES

# In[11]:


# replacing the question mark by the pandas native missing values
adult=adult.replace('?', np.nan)


# In[12]:


# removing the duplicates
adult.drop_duplicates(inplace=True)
# dropping the education column
adult.drop('educational-num', axis=1,inplace=True)
adult.shape


# In[13]:


# changing the outcome variable into binary 
adult['income']=np.where(adult['income']=='<=50K',0,1)
adult['income'].head()


# In[14]:


# Focusing the US therefore all other countries
adult['native-country']=[1 if country=='United-States'else 0 for country in adult['native-country']]


# In[15]:


X=adult.drop('income',1)
y=adult['income']


# In[16]:


to_dum = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race','gender','native-country']


# In[17]:


def to_dummy_df(df, to_dum):
    for x in to_dum:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)
        df = df.drop(x, 1)
        df = pd.concat([df, dummies], axis=1)
    return df


# In[18]:


X = to_dummy_df(X, to_dum)
print(X.head(3))


# In[19]:


#dealing with missing values by Imputation
from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN', strategy='median', axis=0)
imp.fit(X)
X = pd.DataFrame(data=imp.transform(X) , columns=X.columns)


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[21]:


from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier


models = [
    LogisticRegression(),
    KNeighborsClassifier(5),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    XGBClassifier(),
    ]


# In[22]:


log_cols=["Classifier", "Accuracy", "Log Loss"]
#log = pd.DataFrame(columns=log_cols)

for clf in models:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print("="*25)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    print('')
    print(classification_report(y_test,train_predictions))
    
    train_predictions = clf.predict_proba(X_test)
    ll = log_loss(y_test, train_predictions)
    y_hat=[x[1] for x in clf.predict_proba(X_test)]
    score=roc_auc_score(y_test,y_hat)
    print("ROC score: {}".format(score))
    print("Log Loss: {}".format(ll))
    


# Most of models are peforming Okay-ish with only the KNNClassifier performing the established benchmark.
# They could be few ways to improve what we have here:.<br>
# 
# 1- Using GridSearch to tune the best models out there.<br>
# 2- Finding an **n** interations between features that increases the number of features but that could reduced by using using the PCA. (This could also be sorted by using a regularization or just using the SelectBest)<br>
# 3- Balance the classes using SMOTE techniques

# In[23]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


# In[25]:


for clf in models:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.3%}".format(acc))
    print('')
    print(classification_report(y_test,train_predictions))
    
    prob_score = clf.predict_proba(X_test)
    y_hat = [x[1] for x in clf.predict_proba(X_test)]
    ll = log_loss(y_test, train_predictions)
    score=roc_auc_score(y_test,y_hat)
    print("ROC Score: {}".format(score))
    print("Log: {}".format(ll))


# # MODEL IMPROVEMENT
# **Logistic Regression (with Balance-Weight)**<br>
# Due to the symetric nature of the Logistic loss function, we have the class 1 penalised.  Most of these models are not really good despite the ROC-score and accuracy being just over acceptable especially for AdaBoost and XGBoost classifiers that have a ROC-score over 90%.<br> The classification report gives us a good insight on the how the model perform when the the class 1 occurs and that momment it is not very encouraging.<br> Once way to round this problem using **scikit-Learn** is to balance the weights of the classes.
# 

# In[26]:


from sklearn.model_selection import GridSearchCV
logit=LogisticRegression()
grid_values = {'penalty': ['l1', 'l2'],'C':[0.001,.009,0.01,.09,1,5,10,25],'class_weight':['balanced']}


grid_model=GridSearchCV(logit,grid_values,cv=5)
grid_model.fit(X_train,y_train)


# In[27]:


grid_pred=grid_model.predict(X_test)
grid_pred_prob=[x[1] for x in grid_model.predict_proba(X_test)]

print('Accuracy:',accuracy_score(y_test,grid_pred))
print('AUC ROC SCORE:',roc_auc_score(y_test,grid_pred_prob))
print('')
print(classification_report(y_test,grid_pred))


# **RESULTS INTERPRETATION & IMBLEARN**

# With a recall of 83% and a F1-score better over 60% when the event occurs. I like this model better than others.
# They are also few others way to deals with imblanced classes on the outcome variables and this is why **IMBLEARN** was invented.<br>
# Here are few algorithms to deals with this kind of problems:<br>
# 1- RandomUndersampler- from UnderSampler (This usually achieve similar performance to Logistic Regression with balanced weights)<br>
# 2- ENN<br>
# 3- SMOTE<br>
# 4- SMOTE ENN<br>
# 5- Tomek Link Removal<br>
# 6- SMOTE Tomek Link Removal<br>

# # IMBLEARN
# <br>
# In this Tutorial only SMOTE + ENN will be demosntrated. You could try over techniques yourself 

# In[28]:


from imblearn.combine import SMOTEENN
smteen=SMOTEENN(0.8,random_state=42)


# In[29]:


# classes count before re-sampling
y_train.value_counts()


# In[30]:


X_train_res, y_train_res=smteen.fit_sample(X_train,y_train)


# In[31]:


unique, counts=np.unique(y_train_res, return_counts=True)
# class count after re-sampling 
print (np.asarray((unique, counts)).T)


# In[32]:


cv=KFold(n_splits=5, shuffle=True,random_state=42)

new_grid=GridSearchCV(logit,grid_values,cv=cv)
new_grid.fit(X_train_res,y_train_res)

new_grid_pred=grid_model.predict(X_test)
new_grid_prob=[x[1] for x in grid_model.predict_proba(X_test)]

print('Accuracy:',accuracy_score(y_test,new_grid_pred))
print('AUC ROC SCORE:',roc_auc_score(y_test,new_grid_prob))
print('')
print(classification_report(y_test,new_grid_pred))


# This part has some of the codes commeneted because it takes too much time to run as you can imagine.
# <br>You can play with it and tell me the results :). Another version of that file is available with feature interaction.<br>

# In[38]:


# Let's now do it using the XGBoost Classifier
xgb=XGBClassifier(random_state=42,objective='binary:logistic')

params={'n_estimators':[100,200,300],'learning_rate':[0.01,0.05,0.1],
        
       'gamma':[0.01,0.05,0.1], 
        
        'booster':['gbtree','gblinear', 'dart'],
        
        'max_depth': [3, 4, 5],
        
        'gamma': [0.5, 1, 1.5, 2],
        
        'colsample_bytree': [0.6, 0.8, 1.0]
       }
        
#xgb_grid=GridSearchCV(xgb,params,cv=5,n_jobs=-1)
#xgb_grid.fit(X_train_res,y_train_res)


# In[ ]:


#xgb_pred=xgb_grid.predict(X_test)

#xg_loss=[x[1] for x in adboost_grid.xgb_pred(X_test)]

print('Accuracy:',accuracy_score(y_test,xgb_pred))
print('AUC ROC SCORE:',roc_auc_score(y_test,xg_loss))
print('')
print(classification_report(y_test,xgb_pred))


# In[ ]:




