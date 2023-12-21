#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# In[2]:


credit_data =pd.read_csv(r"C:\Users\btdjf\Desktop\Ironhack 2\Mid-bootcamp Project\Mid-bootcamp-Project\Files\german_credit_data.csv")
credit_data


# In[3]:


credit_data.shape


# In[4]:


credit_data.info()


# In[5]:


credit_data.isna().sum()


# In[6]:


credit_data.duplicated().sum()


# In[7]:


credit_data.columns


# In[8]:


credit_data = credit_data.drop('Unnamed: 0', axis=1)


# In[9]:


credit_data


# In[10]:


cols = []
for i in range(len(credit_data.columns)):
    cols.append(credit_data.columns[i].lower().replace(' ', '_'))
credit_data.columns = cols
credit_data


# In[11]:


credit_data['age'].unique()


# In[12]:


credit_data['sex'].unique()


# In[13]:


credit_data['job'].unique()


# In[14]:


credit_data['housing'].unique()


# In[15]:


credit_data['saving_accounts'].unique()


# In[16]:


credit_data['checking_account'].unique()


# In[17]:


credit_data['credit_amount'].unique()


# In[18]:


credit_data['duration'].unique()


# In[19]:


credit_data['purpose'].unique()


# In[20]:


credit_data['risk'].unique()


# In[21]:


credit_data['age'].value_counts()


# In[22]:


credit_data['sex'].value_counts()


# In[23]:


credit_data['job'].value_counts()


# In[24]:


credit_data['housing'].value_counts()


# In[25]:


credit_data['saving_accounts'].value_counts()


# In[26]:


credit_data['checking_account'].value_counts()


# In[27]:


credit_data['credit_amount'].value_counts()


# In[28]:


credit_data['duration'].value_counts()


# In[29]:


credit_data['purpose'].value_counts()


# In[30]:


credit_data['risk'].value_counts()


# In[31]:


credit_data.nunique()


# In[32]:


credit_data["saving_accounts"] = credit_data["saving_accounts"].fillna("none")


# In[33]:


credit_data["checking_account"] = credit_data["checking_account"].fillna("none")


# In[34]:


credit_data


# In[ ]:





# In[35]:


type_jobs = {0: 'unskilled/unemployed/non-resident', 1: 'unskilled-resident', 2: 'skilled employee/official', 3: 'highly skilled employe/employer'}
credit_data['job'] = credit_data['job'].map(type_jobs)


# In[36]:


credit_data


# In[37]:


credit_data.describe().T


# In[38]:


numerical = credit_data.select_dtypes(include=['int64'])
numerical


# In[39]:


categorical = credit_data.select_dtypes('object')
categorical


# In[40]:


credit_data.groupby(['sex', 'risk']).size().unstack()


# In[41]:


credit_data.groupby(['sex', 'risk']).size().unstack().plot(kind='bar', figsize=(12, 6))
plt.title('Risk by Gender')
plt.xlabel('Sex')
plt.show()


# In[42]:


credit_data.groupby(['job', 'risk']).size().unstack()


# In[43]:


credit_data.groupby(['job', 'risk']).size().unstack().plot(kind='bar', figsize=(12, 6))
plt.title('Risk by Job Type')
plt.xlabel('Job Type')
plt.show()


# In[44]:


credit_data.groupby(['housing', 'risk']).size().unstack()


# In[45]:


credit_data.groupby(['housing', 'risk']).size().unstack().plot(kind='bar', figsize=(12, 6))
plt.title('Risk by Housing')
plt.xlabel('Housing Type')
plt.show()


# In[46]:


credit_data.groupby(['checking_account', 'risk']).size().unstack()


# In[47]:


credit_data.groupby(['checking_account', 'risk']).size().unstack().plot(kind='bar', figsize=(12, 6))
plt.title('Risk by Checking Account')
plt.xlabel('Checking Account',)
plt.show()


# In[48]:


credit_data.groupby(['saving_accounts', 'risk']).size().unstack()


# In[49]:


credit_data.groupby(['saving_accounts', 'risk']).size().unstack().plot(kind='bar', figsize=(12, 6))
plt.title('Risk by Saving Account')
plt.xlabel('Saving Account')
plt.show()


# In[50]:


credit_data.groupby(['purpose', 'risk']).size().unstack()


# In[51]:


credit_data.groupby(['purpose', 'risk']).size().unstack().plot(kind='bar', figsize=(16, 6))
plt.title('Risk by Purpose')
plt.xlabel('Purpose')
plt.show()


# In[52]:


credit_data.groupby(['age', 'risk']).size().unstack()


# In[53]:


credit_data.groupby(['age', 'risk']).size().unstack().plot(kind='bar', figsize=(16, 6))
plt.title('Risk by Age')
plt.xlabel('Age')
plt.show()


# In[54]:


credit_data.groupby(['duration', 'risk']).size().unstack()


# In[55]:


credit_data.groupby(['duration', 'risk']).size().unstack().plot(kind='bar', figsize=(16, 6))
plt.title('Risk by Duration')
plt.xlabel('Duration')
plt.show()


# In[56]:


credit_data


# In[57]:


numerical


# In[58]:


categorical


# In[59]:


from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()

columns_to_scale = ['age', 'credit_amount', 'duration']


numerical[columns_to_scale] = min_max_scaler.fit_transform(numerical[columns_to_scale])


# In[60]:


label_encoder = LabelEncoder()

categorical['sex_encoded'] = label_encoder.fit_transform(categorical['sex'])
categorical['housing_encoded'] = label_encoder.fit_transform(categorical['housing'])
categorical['saving_encoded'] = label_encoder.fit_transform(categorical['saving_accounts'])
categorical['checking_encoded'] = label_encoder.fit_transform(categorical['checking_account'])
categorical['purpose_encoded'] = label_encoder.fit_transform(categorical['purpose'])
categorical['risk_encoded'] = label_encoder.fit_transform(categorical['risk'])
categorical['job_encoded'] = label_encoder.fit_transform(categorical['job'])


# In[61]:


columns_to_eliminate = ['sex', 'housing', 'saving_accounts', 'checking_account', 'purpose', 'job', 'risk']
categorical = categorical.drop(columns=columns_to_eliminate)
categorical


# In[62]:


numerical


# In[63]:


sns.heatmap(numerical.corr(),annot=True)
plt.show()


# In[64]:


sns.heatmap(categorical.corr(),annot=True)
plt.show()


# In[65]:


credit_data


# In[66]:


credit_data_2 = pd.concat([numerical, categorical], axis=1)
credit_data_2


# In[67]:


sns.heatmap(credit_data_2.corr(),annot=True)
plt.show()


# In[68]:


X = credit_data_2.drop ('risk_encoded' , axis = 1)
y = credit_data_2['risk_encoded']

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size = 0.2)

LR = LogisticRegression()
LR.fit(X_train, y_train)

LR.score(X_test, y_test)

accuracy = LR.score(X_test, y_test)
print("Accuracy:", accuracy)


# In[69]:


from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score 
from sklearn.metrics import classification_report 

 

pred = LR.predict(X_test)  
pred

print("Precison is: ",precision_score(y_test, pred))
print("Recall is: ",recall_score(y_test, pred))
print("F1 is: ",f1_score(y_test, pred))

print(classification_report(y_test, pred))


# In[70]:


from sklearn.metrics import confusion_matrix 

confusion_matrix(y_test, pred)


# In[71]:


predict_prob = LR.predict_proba(X_test)[0:,1]
predict_prob


threshold = 0.6 
costum_pred = (predict_prob >= threshold).astype(int)
costum_pred


# In[72]:


credit_data


# In[73]:


credit_data_2


# In[74]:


from sklearn.neighbors import KNeighborsClassifier


# In[75]:


model = KNeighborsClassifier(n_neighbors = 10)
model = model.fit(X_train, y_train)
pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)


# In[76]:


print("precision: ",precision_score(y_test,pred))
print("recall: ",recall_score(y_test,pred))
print("f1: ",f1_score(y_test,pred))


print(classification_report(y_test, pred))


# In[81]:


from sklearn.metrics import confusion_matrix 

confusion_matrix(y_test, pred)


# In[77]:


credit_data.to_excel('output.xlsx', index=False)


# In[78]:


credit_data_2.to_excel('output_2.xlsx', index=False)


# In[79]:


categorical.to_excel('output_cat.xlsx', index=False)


# In[80]:


numerical.to_excel('output_num.xlsx', index=False)


# In[ ]:




