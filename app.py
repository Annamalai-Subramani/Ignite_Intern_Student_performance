import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the saved model
model = joblib.load('D:/Projects/Ignite_Intern/Student_performance/logistic_regression_model.joblib')

# Function to preprocess data
def preprocess_data(df):
    # Apply Label Encoding to categorical features
    categorical_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
    label_encoders = {}
    for feature in categorical_features:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
        label_encoders[feature] = le

  

    return df

# Streamlit app
def main():
    st.title('Student Performance Prediction')

    # Input fields for user to enter data
    gender = st.selectbox('Gender', ['female', 'male'])
    race_ethnicity = st.selectbox('Race/Ethnicity', ['group A', 'group B', 'group C', 'group D', 'group E'])
    parental_education = st.selectbox('Parental Level of Education', ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"])
    lunch = st.selectbox('Lunch', ['standard', 'free/reduced'])
    test_preparation = st.selectbox('Test Preparation Course', ['none', 'completed'])
    math_score = st.slider('Math Score', min_value=0, max_value=100, value=50)
    reading_score = st.slider('Reading Score', min_value=0, max_value=100, value=50)
    writing_score = st.slider('Writing Score', min_value=0, max_value=100, value=50)

    # Predict button
    if st.button('Predict'):
        # Create DataFrame with user input
        input_data = {
            'gender': [gender],
            'race/ethnicity': [race_ethnicity],
            'parental level of education': [parental_education],
            'lunch': [lunch],
            'test preparation course': [test_preparation],
            'math score': [math_score],
            'reading score': [reading_score],
            'writing score': [writing_score]
        }
        df = pd.DataFrame(input_data)

        # Preprocess the input data
        df_processed = preprocess_data(df)

        # Make predictions
        prediction = model.predict(df_processed)
        prediction_proba = model.predict_proba(df_processed)

        # Display prediction
        st.subheader('Prediction')
        if prediction[0] == 1:
            st.write('The student is predicted to pass (average score >= 70).')
        else:
            st.write('The student is predicted to fail (average score < 70).')

        # Display prediction probabilities if available
        if prediction_proba is not None:
            st.subheader('Prediction Probabilities')
            st.write(f'Probability of failing (average score < 70): {prediction_proba[0][0]:.2f}')
            st.write(f'Probability of passing (average score >= 70): {prediction_proba[0][1]:.2f}')

# Run the app
if __name__ == '__main__':
    main()
