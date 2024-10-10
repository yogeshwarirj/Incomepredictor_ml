# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 16:30:02 2024

@author: rames
"""
import pandas as pd
import numpy as np
#import random
#from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
#import pickle
from ucimlrepo import fetch_ucirepo
import joblib

# Fetch the Adult dataset
dataset = fetch_ucirepo('adult')
#print(dataset)
# Access data 
actual_data=dataset.data.features # give dataset except target
target = dataset.data.targets
# Convert to pandas DataFrame
df1 = pd.DataFrame(data=actual_data, columns=dataset.headers)
df1["income"]=target

df1['income']=df1['income'].replace('<=50K.','<=50K')
df1['income']=df1['income'].replace('>50K.','>50K')

# Define bins and labels for age categories
bins = [0, 13, 20, 35, 61, 100]  # Age intervals
labels = ['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']  # Corresponding labels

# Use pd.cut to categorize the age into groups
df1['age_group'] = pd.cut(df1['age'], bins=bins, labels=labels, right=False)

# Function to categorize grade levels
def categorize_grade(grade):
    if grade in ['Preschool', '1st-4th']:
        return 'Elementary'
    elif grade in ['5th-6th', '7th-8th']:
        return 'Middle School'
    elif grade in ['9th', '10th', '11th', '12th']:
        return 'Not HS-grad'
    else:
        return grade
        
# Apply the function to categorize grade levels
df1['education_category'] = df1['education'].apply(categorize_grade)

df1['workclass'].replace(['?', ' '], np.nan, inplace=True)
df1['occupation'].replace(['?', ' '], np.nan, inplace=True)
df1['native-country'].replace(['?', ' '], np.nan, inplace=True)

df1.loc[df1['workclass'] == 'Never-worked', 'hours-per-week'] = 0

df1 = df1.drop_duplicates()
# dropping the unnecessary columns: fnlwgt(as it is a cencus adjustment value), education(as we have education number)
columns_to_drop = ['fnlwgt', 'education','capital-gain', 'capital-loss', 'education-num','age','workclass'] 
#Dropping the specified columns
df1 = df1.drop(columns=columns_to_drop)

#from sklearn.preprocessing import LabelEncoder

# columns that we are encoding: 
columns_to_encode = ['age_group', 'marital-status', 'occupation', 'relationship','race', 
                     'sex', 'native-country', 'income' ,'education_category'] 
# Initialize the LabelEncoder
le = LabelEncoder()

# Dictionary to store mappings for each column
label_mappings = {}

# Apply label encoding and save mappings
for col in columns_to_encode:
    df1[col] = le.fit_transform(df1[col])
    label_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

# Display the mapping of each encoded column
for col, mapping in label_mappings.items():
    print(f"Mapping for {col}: {mapping}")
    
    
    
    
    #from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import train_test_split



# Separate rows with and without missing values in 'occupation', 14 is mapping to null
df_missing =  df1[df1['occupation'] == 14]
df_not_missing =  df1[df1['occupation'] != 14]


# imputing missing values in occupation
# Define features and target
X =  df_not_missing.drop(columns=['occupation'])   # Features
y = df_not_missing['occupation'] # Target variable (occupation)


# Split the non-missing data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Use the trained model to predict the 'workclass' for rows with missing values
X_missing = df_missing.drop(columns=['occupation'])  # Features for rows with missing values
df_missing['occupation'] = rf.predict(X_missing)

# Combine the data back together
df_imputed = pd.concat([df_not_missing, df_missing])

# Display the imputed DataFrame
#print(df_imputed)

df1=df_imputed.copy()

#Since hours-per-week, native country and race has VIF > 10 indicating high colinearity,
#we can keep hours-per-week and remove the other two as hours per week is more meaningful feature for the salary prediction(to this domain). 
df1  = df1.drop(columns = ['native-country', 'race'])

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
# Assuming 'income' is the target variable with values '>50K' and '<=50K'
X = df1.drop(columns=['income'])  # Features
y = df1['income']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Apply SMOTE to the training data
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check the distribution after oversampling
#print("Before SMOTE:", y_train.value_counts())
#print("After SMOTE:", pd.Series(y_train_resampled).value_counts())

# Convert the resampled data back into a DataFrame
X_resampled_df = pd.DataFrame(X_train_resampled, columns=X.columns)
y_resampled_df = pd.DataFrame(y_train_resampled, columns=['income'])

# Combine the features and target into a single DataFrame
df_resampled = pd.concat([X_resampled_df, y_resampled_df], axis=1)
df1=df_resampled.copy()
X = df1.drop(columns=['income'])  # Features
y = df1['income']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Saving model to disk
#pickle.dump(model, open('model.pkl','wb'))
# Save the model using joblib
joblib.dump(model, 'model.joblib')
# Loading model to compare the results
#model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[2, 9, 6]]))




