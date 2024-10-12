
import numpy as np

import streamlit as st
import pickle
# Load the trained model (same as Flask)
model = pickle.load(open('model.pkl', 'rb'))
# Load the saved model
#import joblib
#model = joblib.load('model.joblib')

# Create mappings (same as Flask)
age_groupm = {'Adult': 0, 'Senior': 1, 'Teen': 2, 'Young Adult': 3}
marital_statusm = {'Divorced': 0, 'Married-AF-spouse': 1, 'Married-civ-spouse': 2,
                   'Married-spouse-absent': 3, 'Never-married': 4,
                   'Separated': 5, 'Widowed': 6}
occupation_mappingm = {'Adm-clerical': 0, 'Armed-Forces': 1, 'Craft-repair': 2,
                       'Exec-managerial': 3, 'Farming-fishing': 4, 'Handlers-cleaners': 5,
                       'Machine-op-inspct': 6, 'Other-service': 7, 'Priv-house-serv': 8,
                       'Prof-specialty': 9, 'Protective-serv': 10, 'Sales': 11,
                       'Tech-support': 12, 'Transport-moving': 13}
relationshipm = {'Husband': 0, 'Not-in-family': 1, 'Other-relative': 2, 'Own-child': 3,
                 'Unmarried': 4, 'Wife': 5}
sexm = {'Female': 0, 'Male': 1}
education_categorym = {'Assoc-acdm': 0, 'Assoc-voc': 1, 'Bachelors': 2, 'Doctorate': 3,
                       'Elementary': 4, 'HS-grad': 5, 'Masters': 6, 'Middle School': 7,
                       'Not HS-grad': 8, 'Prof-school': 9, 'Some-college': 10}

# Function to categorize age
def categorize_age(age):
    if 0 <= age <= 12:
        return 'Child'
    elif 13 <= age <= 19:
        return 'Teen'
    elif 20 <= age <= 34:
        return 'Young Adult'
    elif 35 <= age <= 60:
        return 'Adult'
    else:
        return 'Senior'

# Function to categorize education level
def categorize_grade(grade):
    if grade in ['Preschool', '1st-4th']:
        return 'Elementary'
    elif grade in ['5th-6th', '7th-8th']:
        return 'Middle School'
    elif grade in ['9th', '10th', '11th', '12th']:
        return 'Not HS-grad'
    else:
        return grade

# Function to handle outliers in hours per week
def outlier_hours_per_week(hr):
    if hr >= 80:
        return 80
    else:
        return hr

# Streamlit app
st.title("Income Prediction App")

# Input form in Streamlit
with st.form("prediction_form"):
    hours_per_week = st.number_input('Hours per week', min_value=1, max_value=100)
    age_group = st.number_input('Age', min_value=0, max_value=100)
    sex = st.selectbox('Sex', ['Male', 'Female'])
    marital_status = st.selectbox('Marital Status', list(marital_statusm.keys()))
    occupation = st.selectbox('Occupation', list(occupation_mappingm.keys()))
    relationship = st.selectbox('Relationship', list(relationshipm.keys()))
    education_category = st.selectbox('Education Level', list(education_categorym.keys()))
    
    submit_button = st.form_submit_button(label='Predict')

# Handle form submission
if submit_button:
    # Handling outliers and categorizing input
    hours_per_week = outlier_hours_per_week(hours_per_week)
    age_group_categorized = categorize_age(age_group)
    education_category_categorized = categorize_grade(education_category)

    # Convert to encoded values
    age_groupe = age_groupm[age_group_categorized]
    marital_statuse = marital_statusm[marital_status]
    occupatione = occupation_mappingm[occupation]
    relationshipe = relationshipm[relationship]
    education_categorye = education_categorym[education_category_categorized]
    sexe = sexm[sex]

    # Prepare the feature array for prediction
    final_features = np.array([hours_per_week, age_groupe, marital_statuse, occupatione,
                               relationshipe, education_categorye, sexe]).reshape(1, -1)

    # Make the prediction
    prediction = model.predict(final_features)

    # Display the result
    if prediction[0] == 0:
        st.success('Income of this person is <=50K')
    else:
        st.success('Income of this person is >50K')
