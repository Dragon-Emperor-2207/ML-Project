#!/usr/bin/env python
# coding: utf-8

# In[80]:


#importing the relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


# In[81]:


import sklearn
sklearn.__version__


# In[82]:


# to ignore all the warnings 
# import warnings
# warnings.filterwarnings("ignore")


# In[83]:


loan_approval = pd.read_csv(r"C:\Users\harsh\Downloads\train_u6lujuX_CVtuZ9i.csv");
loan_approval.head()


# In[84]:


# Creating the feature MAtrix
X = loan_approval.drop(['Loan_Status'],axis=1)

# Creating the label matrix
y = loan_approval['Loan_Status']


# ## Getting our data ready
# 
# 1. split the training data into train and test splits
# 2. Convert non-numerical values into numerical values
# 3. Imputing/Disregarding the missing values

# In[85]:


# Spliting the data into training and test splits
np.random.seed(42)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2) ## test_size=0.2 means that 20% data will be used for testing and 80% data will be used for training
X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[86]:


# Checking for datatypes 
loan_approval.dtypes


# In[87]:


# We are very sure that loan_ID isn't a determining factor in wheter a person will get a loan or not
# so we drop the Loan_ID column from the dataframe

loan_approval.drop(['Loan_ID'],axis=1,inplace=True)
loan_approval


# In[88]:


# We only want to deal with numerical data , so convert the non numerical data into numerical data

# Turn the categories into numbers
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_features = ["Gender","Married","Education","Self_Employed","Property_Area"]
one_hot = OneHotEncoder()

transformer = ColumnTransformer([("one_hot",
                                   one_hot,
                                   categorical_features)],
                                   remainder="passthrough")

transformed_X = transformer.fit_transform(X)
transformed_X



# In[89]:


np.random.seed(42)
X_train,X_test,y_train,y_test = train_test_split(transformed_X,
                                                   y,
                                                   test_size=0.2)


# ##  Dealing with missing value
# 
# 1. Either fill the missing value with a suitable value
# 2. Else remove the samples from the dataset altogether

# In[90]:


loan_approval.isna().sum()


# In[91]:


loan_approval.dtypes


# In[92]:


transformed_X


# In[93]:


# We will fill the missing values using sci kit learn


# In[94]:


# Drop the row with no labels
loan_approval.dropna(subset=['Loan_Status'],inplace=True)
loan_approval.isna().sum()


# In[95]:


#Split into X and y
X = loan_approval.drop(['Loan_Status'],axis=1)
y = loan_approval['Loan_Status']


# In[96]:


X.shape,y.shape


# In[99]:


# FIll missing values with sci-kit learn
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Fill categorical values with "missing" and numerical values with mean

cat_imputer = SimpleImputer(strategy="constant",fill_value="missing") #cat meaning categoricalfilconst_imputer = SimpleImputer(strategy="constant",fill_value='3')
num_imputer = SimpleImputer(strategy="mean")

# Define features
cat_features = ["Gender","Married","Self_Employed","Education",'Dependents','Property_Area']
num_features = ['Credit_History','Loan_Amount_Term','LoanAmount','ApplicantIncome','CoapplicantIncome']

# Create an imputer (something that fills missing data)
imputer = ColumnTransformer([
    ("cat_imputer",cat_imputer,cat_features),
    ("num_imputer",num_imputer,num_features)
])

#Transform the data
filled_X = imputer.fit_transform(X)
filled_X


# In[100]:


filled_X.shape


# In[101]:


loan_approval


# In[102]:


loan_approval_filled = pd.DataFrame(filled_X,columns=["Gender","Married","Self_Employed","Education",'Dependents','Property_Area','Credit_History','Loan_Amount_Term','LoanAmount','ApplicantIncome','CoapplicantIncome'])
loan_approval_filled.head()


# In[103]:


loan_approval_filled.isna().sum()


# In[104]:


X = loan_approval_filled
y = loan_approval['Loan_Status']


# In[128]:


X.shape


# In[129]:


# Converting the categorical data into numbers
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_features = ["Gender", "Married", "Self_Employed", "Education", "Dependents", "Property_Area"]
one_hot = OneHotEncoder()

transformer = ColumnTransformer([("one_hot", one_hot, categorical_features)], remainder="passthrough")

transformed_X = transformer.fit_transform(X)
print(transformed_X.shape)


# In[127]:


transformed_X.shape


# In[120]:


y.dtype


# In[121]:


loan_approval['Loan_Status'] = loan_approval['Loan_Status'].replace({'Y': 1, 'N': 0})
y = loan_approval['Loan_Status']


# In[116]:


y


# In[138]:


transformed_X.shape


# ## Fitting our model
# 
# 

# In[132]:


# now we have our data in numbers and with filled(with no missing values)
# Lets fit a model
np.random.seed(42)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
model = RandomForestClassifier()
X_train,X_test,y_train,y_test = train_test_split(transformed_X,y,test_size=0.2)
model.fit(X_train,y_train) # will find the pattern between X_train and y_train
model.score(X_train,y_train)


# In[134]:


# import thr linear SVC estimator class
from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split

#Setup random seed
np.random.seed(42)

# make the data
X = transformed_X


#split the data into training and testing
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# instantiate LinearSVC
clf = LinearSVC(max_iter=10000)
clf.fit(X_train,y_train) 

# Evaluate the linear SVC
clf.score(X_test,y_test)


# In[140]:


# import thr linear RandomForestClassifier estimator class
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

#Setup random seed
np.random.seed(42)

# make the data
X = transformed_X

#split the data into training and testing
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# instantiate RandomForestClassifier
clf = RandomForestClassifier()

# Fit the model to the data (training the machine learning model)
clf.fit(X_train,y_train) 

# Evaluate the RandomForestClassifier (use the patterns that the model has learnt)
clf.score(X_test,y_test)


# In[143]:


from sklearn.metrics import mean_absolute_error
y_preds = clf.predict(X_test)

mean_absolute_error(y_preds,y_test)


# ### Since our model is doing better in Random Forest Classifier , we will use RandomForestClassifier for the time being

# ## Evaluating our model
# 

# In[148]:


## Evaluating a model with score() method 
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

X = transformed_X

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

clf = RandomForestClassifier()

clf.fit(X_train,y_train)


# In[149]:


clf.score(X_train,y_train)


# In[150]:


clf.score(X_test,y_test)


# In[151]:


cross_val_score(clf,X,y,cv=5)
# these scores are representing 


# ### Classification model evaluation matrix
# 
# 1. Accuracy
# 2. Area under ROC curve
# 3. Confusion matrix
# 4. Classification report
# 

# In[152]:


# 1. Accuracy 


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

X = transformed_X

clf = RandomForestClassifier()
cross_val_scorex = cross_val_score(clf,X,y,cv=5)


# In[153]:


np.mean(cross_val_scorex)


# **Area under the receiver operating characterstic curve(AUC/ROC)**
# 
# * Area under curve (AUC)
# * ROC
# 
# ROC curves are a comparison of a model's true positive rate (tpr) versus a model's false positive rate (fpr)
# 
# * True positive = model predicts 1 when truth is 1
# * False positive = model predicts 1 when truth is 0
# * True negative = model predicts 0 when truth is 0
# * False negative = model predicts 0 when truth is 1

# In[154]:


# Create X_test... etc
X_train,X_test,y_train,y_test = train_test_split(transformed_X,y,test_size=0.2)


# In[155]:


from sklearn.metrics import roc_curve

# Fit the classifier
clf.fit(X_train,y_train)

# make predictions with probabilities
y_probs =  clf.predict_proba(X_test)

y_probs[:10]



# In[156]:


y_probs_positive = y_probs[:,1]
y_probs_positive[:10]


# In[157]:


# calculate fpr, tpr and thresholds
fpr,tpr,thresholds = roc_curve(y_test,y_probs_positive)

#check the false positive rates
fpr


# In[158]:


# create a function for plotting ROC curves

import matplotlib.pyplot as plt 

def plot_roc_curve(fpr,tpr):
    '''
    Plots the ROC curve given the false positive rate(fpr) and true positive rate(tpr) of a model
    '''
    #plot roc_curve
    plt.plot(fpr,tpr,color="orange",label="ROC")
    #plot line with no predictive power (baseline)
    plt.plot([0,1],[0,1],linestyle='--',color='darkblue')
    
    #Customise the plot 
    plt.xlabel("False positive rate (fpr)")
    plt.ylabel("True positive rate (tpr)")
    plt.title("Receiver Operating Characterstics (ROC) curve")
    plt.legend()
    plt.show()
    
    
plot_roc_curve(fpr,tpr)


# In[159]:


from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_probs_positive)


# In[160]:


# Plot perfect ROC curve and AUC score
fpr,tpr,thresholds = roc_curve(y_test,y_test)
plot_roc_curve(fpr,tpr)

#perfect ROC score
roc_auc_score(y_test,y_test)


# **Confusion Matrix**
# 
# A confusion matrix compares the label of a model predicts and the actual labels it was supposed to predict
# 
# In essence, it gives you an idea of where the model is getting confused

# In[161]:


from sklearn.metrics import confusion_matrix
y_preds = clf.predict(X_test)
confusion_matrix(y_test,y_preds)


# In[162]:


# Visualise confusion matrix wirh pd.crosstab()

pd.crosstab(y_test,
              y_preds,
              rownames=["Actual label"],
              colnames=["Predicted labels"])

#values on the diagonal are predicted right


# In[163]:


# make our confusion matrix more visual with seaborn's heatmap()
import seaborn as sns

#set the font scale
sns.set(font_scale=1.5)

# Create a confusion matrix
conf_mat = confusion_matrix(y_test,y_preds)

#plot it using seaborn 
sns.heatmap(conf_mat);


# In[165]:


from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_true=y_test,
                                        y_pred=y_preds);


# ## Classification Report

# 1) **Accuracy** is a good measure to start with if all classes are balanced (eg: same amount of samples which are labelled as 0 and 1).
#     
# 2) **Precision** and **recall** become more important when classes are imbalanced.
# 
# 3) If the false positive predictions are worse than false negatives, aim for higher precision
# 
# 4) If false negative predictions are worse than false positives , aim for higher recall.
# 
# 5) **F1-score** is a combination of precision and recall\

# In[167]:


from sklearn.metrics import classification_report

print(classification_report(y_test,y_preds))


# In[168]:


from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
np.random.seed(42)

#Create X and y
X = transformed_X
# Split the data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# Fit the model
clf = RandomForestClassifier()

#Fit model
clf.fit(X_train,y_train)

# Make predictions 
y_preds = clf.predict(X_test)

# Evaluating the function based on evaluation functions
print("Classifier metrics on the test set")
print(f"Accuracy : {accuracy_score(y_test,clf.predict(X_test))*100:.2f}%")
print(f"Accuracy : {accuracy_score(y_test,y_preds)*100:.2f}%")
print(f"Precision : {precision_score(y_test,y_preds)}")
print(f"Recall : {recall_score(y_test,y_preds):.2f}")
print(f"F1 : {f1_score(y_test,y_preds):.2f}")


# In[172]:


filled_df = pd.DataFrame(transformed_X)
filled_df['Loan_Status']=y


# ### Trying to improve a model

# In[174]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()


# In[175]:


clf.get_params()


# In[ ]:




