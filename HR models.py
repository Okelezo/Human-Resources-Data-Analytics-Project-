#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries to be used

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# # Reading data into the working environment

# In[2]:


hr_data = pd.read_csv("https://raw.githubusercontent.com/Okelezo/Human-Resources-Data-Analytics-Project-/main/HRDataset_v9.csv")
pd.set_option('display.max_columns', None)
hr_data.head()


# # EDA

# In[3]:


hr_data.shape


# In[4]:


hr_data.isna().sum()


# In[5]:


hr_data.duplicated().sum()


# In[6]:


employee_plot = hr_data[['Sex', 'MaritalDesc', 'CitizenDesc', 'RaceDesc', 
                         'Department', 'Employee Source', 'Employment Status']]
employee_plot['Employment Status']= (employee_plot['Employment Status']
                                     .apply(lambda x: 1 if x in ['Voluntarily Terminated','Terminated for Cause'] else 0))


# In[7]:


cols = ['Sex', 'MaritalDesc']

fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(20,10))
for col, axes in zip(cols, ax.flatten()):
    plt.style.use('seaborn-darkgrid')
    sns.countplot(data=employee_plot, x=col, hue='Employment Status', ax=axes)
    axes.set_xticklabels(employee_plot[col].unique(), fontsize=15)
    axes.set_title(f'COUNTS OF {col.upper()} ON EMPLOYMENT STATUS', fontsize=20, fontweight='bold')
    axes.set_xlabel(f'{col}', fontsize=17, labelpad=5)
    axes.set_ylabel('Counts', fontsize=17)
    axes.legend(loc=1, title='Employment Status', fontsize=10)
    fig.tight_layout(pad=5);


# In[8]:


cols = ['Employee Source', 'RaceDesc']

fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(20,10))
for col, axes in zip(cols, ax.flatten()):
    plt.style.use('seaborn-darkgrid')
    sns.countplot(data=employee_plot, x=col, hue='Employment Status', ax=axes)
    axes.set_xticklabels(employee_plot[col].unique(), fontsize=15, rotation=90)
    axes.set_title(f'COUNTS OF {col.upper()} ON EMPLOYMENT STATUS', fontsize=20, fontweight='bold')
    axes.set_xlabel(f'{col}', fontsize=17, labelpad=5)
    axes.set_ylabel('Counts', fontsize=17)
    axes.legend(loc=1, title='Employment Status', fontsize=10)
    fig.tight_layout(pad=5);


# In[9]:


cols = ['Department', 'CitizenDesc']

fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(20,10))
for col, axes in zip(cols, ax.flatten()):
    plt.style.use('seaborn-darkgrid')
    sns.countplot(data=employee_plot, x=col, hue='Employment Status', ax=axes)
    axes.set_xticklabels(employee_plot[col].unique(), fontsize=15)
    axes.set_title(f'COUNTS OF {col.upper()} ON EMPLOYMENT STATUS', fontsize=20, fontweight='bold')
    axes.set_xlabel(f'{col}', fontsize=17, labelpad=5)
    axes.set_ylabel('Counts', fontsize=17)
    axes.legend(loc=1, title='Employment Status', fontsize=10)
    fig.tight_layout(pad=5);


# In[10]:


Term = hr_data['Reason For Term'].value_counts()
Term


# In[11]:


Term = Term.drop(['N/A - still employed', 'N/A - Has not started yet'], 0)
Term


# In[12]:


Term.plot(kind='barh', figsize=(15, 8))
plt.ylabel('Reason For Termination', fontsize=15)
plt.xlabel('Counts', fontsize=15)
plt.title('COUNT PLOT OF REASON FOR EMPLOYEE TERMINATION', fontweight='bold', fontsize=20)
for i, value in zip(range(len(Term)), Term.values):
    plt.text(value+0.05, i, '%d'%value, ha='left', va='center', fontsize=10)


# # Relevant features needed to be considered for the model

# > The features selected to be considered and relevant to building the models are;
# 
# * MaritalDesc (emcoded as MaritalStatusID)
# * MarriedID               
# * Sex (encoded as GenderID)
# * Department (encoded as DeptID)
# * Performance Score (encoded as Perf_ScoreID)
# * Age
# * Pay Rate
# * CitizenDesc
# * Hispanic/Latino
# * RaceDesc
# * Days Employed
# * Position
# * Manager Name
# * Employee Source
# > Target variable
# * Employment Status 

# In[13]:


data = hr_data[['MarriedID', 'MaritalStatusID', 'GenderID', 'DeptID', 'Perf_ScoreID',
                'Age', 'Pay Rate', 'Days Employed','CitizenDesc', 'Hispanic/Latino', 'RaceDesc', 
              'Position', 'Manager Name', 'Employee Source','Employment Status']]
data.head()


# In[14]:


data.info()


# In[15]:


data.shape


# # Data Preparation
# 
# ## Encoding features in the relevant data

# > Applying the One Hot Encoder method here, using the `pd.get_dummies()` method<br>
# > Using `ordinal encoder` for the variable *CitizenDesc and Hispanic/Latino*

# In[16]:


from sklearn.preprocessing import OrdinalEncoder


# In[17]:


columns = ['RaceDesc', 'Position', 'Manager Name', 'Employee Source']
data_encoded = pd.get_dummies(data=data, columns=columns)
data_encoded.head()


# In[18]:


data['CitizenDesc'].value_counts()


# In[19]:


data['Hispanic/Latino'].value_counts()


# In[20]:


data.loc[:, 'Hispanic/Latino'] = data['Hispanic/Latino'].apply(lambda x: x.lower())
data['Hispanic/Latino'].value_counts()


# In[21]:


encode = OrdinalEncoder()

ordinal_feat = ['CitizenDesc', 'Hispanic/Latino']
me = encode.fit_transform(data[ordinal_feat])

encode.categories_


# In[22]:


data_encoded[ordinal_feat] = me
for cols in ordinal_feat:
    print(data[cols].value_counts(), '\n')


# In[23]:


data_encoded.shape


# In[24]:


data_encoded.head()


# ## Encoding the target variable

# > Of all classes in the target varible, Employment Status. Our positive class (1) would be `Voluntarily Terminated` and `Terminated for Cause`. The rest would represent negative class (0).

# In[25]:


data_encoded['Employment Status'].unique()


# In[26]:


data_encoded['Employment Status'].value_counts()


# In[27]:


# function to encode the target variables into 1's and 0's as discussed earlier

def target(status):
    if status == 'Voluntarily Terminated' or status == 'Terminated for Cause':
        return 1
    else:
        return 0


# In[28]:


data_encoded['Employment Status'] = data_encoded['Employment Status'].apply(target)
data_encoded['Employment Status'].value_counts()


# In[29]:


data_encoded.head()


# ### Splitting data into features, X and target, y

# In[30]:


X = data_encoded.drop('Employment Status', axis=1)
y = data_encoded['Employment Status']


# In[31]:


X.shape, y.shape


# > After encoding and preparing the data for the model, it has 88 columns and 310 rows.

# ### Feature Scaling

# In[32]:


# importing neccessary modules for the feature scaling

from sklearn.preprocessing import MinMaxScaler, StandardScaler


# In[33]:


# instantiating standard scaler
scaler = StandardScaler()


# ## Models to be used

# In[34]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from mlxtend.classifier import StackingClassifier


# # Metrics to be used

# In[35]:


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


# In[36]:


from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import make_pipeline


# #### Custom function for model training and evaluation

# In[37]:


# Building a fucntion that applies SKF to split data, fit data to pipeline, make predictions and evaluate their performance


def model_train_eval(X, y, splits=5, model=None):
    # using an n_splits of 5 garauntees a 80 - 20% train test split with a balanced class for each iteration.
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

    # create lists to store different metric scores from model at each iteration
    train_score = []
    test_score = []
    precision = []
    recall = []
    accuracy = []
    f1_scores = []
    
    # applying SKFold split on data
    for train_index, test_index in skf.split(X, y):
        X_train = X.loc[train_index,:]
        y_train = y[train_index]
        X_test = X.loc[test_index,:]
        y_test = y[test_index]

        # fit pipeline on data
        # pipeline performs feature selection on data then passes selected features to model
        pipe = make_pipeline(scaler, model)
        pipe.fit(X_train, y_train)
        # score model on train set
        train_score.append(pipe.score(X_train, y_train))
        # predict on X_test set
        y_preds = pipe.predict(X_test)
        # score model on test set
        test_score.append(pipe.score(X_test, y_test))
        
        # appending all metric scores to their respective list place holders
        precision.append(precision_score(y_test, y_preds))
        recall.append(recall_score(y_test, y_preds))
        accuracy.append(accuracy_score(y_test, y_preds))
        f1_scores.append(f1_score(y_test, y_preds))
    
    # averaging all metric scores and printing the values.
    print(f'MODEL USED------>{model}')
    print('===================================================================>')
    print(f'Train score: {np.mean(train_score)} (+/- {np.std(train_score)})\n')
    print(f'Test score: {np.mean(test_score)} (+/- {np.std(test_score)})\n')
    print(f'Precision score: {np.mean(precision)} (+/- {np.std(precision)})\n')
    print(f'Recall score: {np.mean(recall)} (+/- {np.std(recall)})\n')
    print(f'Accuracy score: {np.mean(accuracy)} (+/- {np.std(accuracy)})\n')
    print(f'F1 score: {np.mean(f1_scores)} (+/- {np.std(f1_scores)})')


# Steps taken in the function above;
# > Use StatifiedKFold to split data into train and test splits for 5 folds <br>
# > Pass the train set to pipepline, which scales the data using `StandardScaler()` method and then fits chosen model to scaled data<br>
# > Predicts on test set using the pipeline<br>
# > Evaluate the model with the metrics and stores score in a list.<br>
# 
# Repeats this process for the 5 folds and finally returns the average and standard deviation of all scores

# # Model building
# 
# We will build base models and try to stack them together to improve their performance.
# 
# **`Base Models`**
# > 1. KNeighbors classifier
# > 2. Logistic Regression
# > 3. Decision Trees
# > 4. Random forest classifier
# > 5.Gradient Boosting Classifier
# 
# **`Stacked Models`**<br>
# We would be trying out different combinations of best performing base models and use 2 simple models as meta classifiers to train on the predictions of the base models.
# The two meta classifiers used here are;
# > 1. Decision Trees, and 
# > 2. Logistic Regression

# # Base Models
# 
# ### KNeighborsClassifier as base model

# In[38]:


# instantiating model
knc = KNeighborsClassifier(n_neighbors=5)

# using model_train_eval function to train and evaluate the instantiated model above
model_train_eval(X, y, model=knc)


# ### Logistic Regression as base model

# In[39]:


# instantiating model
log_reg = LogisticRegression(random_state=42, max_iter=200)

# using model_train_eval function to train and evaluate the instantiated model above
model_train_eval(X, y, model=log_reg)


# ### Decision Trees as base model

# In[40]:


# instantiating model
dec_tree = DecisionTreeClassifier(random_state=42)

# using model_train_eval function to train and evaluate the instantiated model above
model_train_eval(X, y, model=dec_tree)


# ### Random Forest as base model

# In[41]:


rfc = RandomForestClassifier(random_state=42)

model_train_eval(X, y, model=rfc)


# ### Gradient Boosting Classifier as base model

# In[42]:


gbc = GradientBoostingClassifier(random_state=42, n_estimators=200)

model_train_eval(X, y, model=gbc)


# # Stacked Models
# 
# ### Decision Trees as meta classifier

# **`Base models:` Decision Trees, Random forest classifier**

# In[43]:


# instantiating the model
stack_model = StackingClassifier(classifiers=[dec_tree, rfc],
                                 use_probas=True,
                                 average_probas=True,
                                 meta_classifier=dec_tree)

# using model_train_eval function to train and evaluate the instantiated model above
model_train_eval(X, y, model=stack_model)


# **`Base models:`Logistic Regression, Decision Trees, Random forest classifier**

# In[44]:


# instantiating the model
stack_model = StackingClassifier(classifiers=[dec_tree, rfc, log_reg],
                                 use_probas=True,
                                 average_probas=True,
                                 meta_classifier=dec_tree)

# using model_train_eval function to train and evaluate the instantiated model above
model_train_eval(X, y, model=stack_model)


# **`Base models:` Logistic Regression, Decision Trees, Random forest classifier, Gradient Boosting Classifier**

# In[45]:


# instantiating the model
stack_model = StackingClassifier(classifiers=[dec_tree, rfc, log_reg, gbc],
                                 use_probas=True,
                                 average_probas=True,
                                 meta_classifier=dec_tree)

# using model_train_eval function to train and evaluate the instantiated model above
model_train_eval(X, y, model=stack_model)


# ## Logistic Regression as meta classifier
# 
# **`Base models:`  Decision Trees, Random forest classifier**

# In[46]:


# instantiating the model
stack_model = StackingClassifier(classifiers=[dec_tree, rfc],
                                 use_probas=True,
                                 average_probas=True,
                                 meta_classifier=log_reg)

# using model_train_eval function to train and evaluate the instantiated model above
model_train_eval(X, y, model=stack_model)


# **`Base models:` Logistic Regression, Decision Trees, Random forest classifier, Gradient Boosting Classifier**

# In[47]:


# instantiating the model
stack_model = StackingClassifier(classifiers=[dec_tree, rfc, log_reg, gbc],
                                 use_probas=True,
                                 average_probas=True,
                                 meta_classifier=log_reg)

# using model_train_eval function to train and evaluate the instantiated model above
model_train_eval(X, y, model=stack_model)


# **`Base models:` Decision Trees, Gradient Boosting Classifier**

# In[48]:


# instantiating the model
stack_model = StackingClassifier(classifiers=[gbc, dec_tree],
                                 use_probas=True,
                                 average_probas=True,
                                 meta_classifier=log_reg)

# using model_train_eval function to train and evaluate the instantiated model above
model_train_eval(X, y, model=stack_model)


# **`Base models:`Logistic Regression, Decision Trees, Gradient Boosting Classifier**

# In[49]:


# instantiating the model
stack_model = StackingClassifier(classifiers=[dec_tree, gbc, log_reg],
                                 use_probas=True,
                                 average_probas=True,
                                 meta_classifier=log_reg)

# using model_train_eval function to train and evaluate the instantiated model above
model_train_eval(X, y, model=stack_model)


# **`Base models:` Logistic Regression, Random forest classifier, Gradient Boosting Classifier**

# In[50]:


# instantiating the model
stack_model = StackingClassifier(classifiers=[gbc, log_reg, rfc],
                                 use_probas=True,
                                 average_probas=True,
                                 meta_classifier=log_reg)

# using model_train_eval function to train and evaluate the instantiated model above
model_train_eval(X, y, model=stack_model)


# **`Base models:` Decision Trees, Random forest classifier, Gradient Boosting Classifier**

# In[51]:


# instantiating the model
stack_model = StackingClassifier(classifiers=[gbc, rfc, dec_tree],
                                 use_probas=True,
                                 average_probas=True,
                                 meta_classifier=log_reg)

# using model_train_eval function to train and evaluate the instantiated model above
model_train_eval(X, y, model=stack_model)

