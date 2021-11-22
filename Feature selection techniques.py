#!/usr/bin/env python
# coding: utf-8

# In[120]:


import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from skfeature.function.similarity_based import fisher_score
import matplotlib.pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel, RFECV, SelectKBest, chi2, f_classif
from sklearn import svm, preprocessing
from sklearn.model_selection import StratifiedKFold
get_ipython().run_line_magic('matplotlib', 'inline')


# In[87]:


df = pd.read_excel(r'C:\Users\Bar\Documents\תואר\תואר שני\תזה\ניתוחים\features_without_outlires.xlsx') #place "r" before the path string to address special character, such as '\'. Don't forget to put the file name at the end of the path + '.xlsx'
df.head(5)
X=df.drop(['Y','Tavg'], axis=1)
Y=df['Y']
X.info()
X


# In[ ]:


#Information Gain


# In[19]:


importances = mutual_info_classif(X,Y)
feat_importances=pd.Series(importances, X.columns[0:len(X.columns)])
feat_importances.plot(kind='barh', color='teal')
plt.show()


# In[ ]:


#Fisher’s Score


# In[44]:


X_np=X.to_numpy()

ranks=fisher_score.fisher_score(X_np,Y)

feat_importances = pd.Series(ranks, X.columns[0:len(X.columns)])
feat_importances.plot(kind='barh', color='teal')
plt.show()


# In[ ]:


#Forward Feature Selection


# In[90]:


lr = LogisticRegression()
ffs=SequentialFeatureSelector (lr,k_features='best', forward=True,n_jobs=-1)
ffs.fit(X,Y)
features=list(ffs.k_feature_names_)
features


# In[ ]:


#Backward Feature Selection


# In[91]:


lr = LogisticRegression()
ffs=SequentialFeatureSelector (lr,k_features='best', forward=False,n_jobs=-1)
ffs.fit(X,Y)
features=list(ffs.k_feature_names_)
features


# In[ ]:


#Recursive Feature Elimination


# In[118]:



# Create the RFE object and compute a cross-validated score.
svc = svm.SVC(probability=True,kernel="linear")

# classifications

rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(5),
              scoring='f1')
rfecv.fit(X, Y)
print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
min_features_to_select = 1  # Minimum number of features to consider
plt.plot(range(min_features_to_select,
               len(rfecv.grid_scores_) + min_features_to_select),
         rfecv.grid_scores_)
plt.show()


# In[126]:


x = X.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
dfnew = pd.DataFrame(x_scaled)

X_new = SelectKBest(f_classif, k=7)
x = X_new.fit(dfnew, Y)
df_new= X.loc[:,x.get_support()]
df_new.info()

