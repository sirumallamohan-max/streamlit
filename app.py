import streamlit as st
import pickle
import pandas as pd

# Load the model
with open('MohanSirumalla_diabetes.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

def main():
    st.title("Diabetes Logistic Regression")

    col1, col2 = st.columns(2)
    #'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age'])
    with col1:
        feature1 = st.slider("Pregnancies", min_value=0, max_value=17,value=6)
        feature2 = st.number_input('Glucose', min_value=0, max_value=199,value=148)
        feature3 = st.number_input('BloodPressure',value=72)
        feature4 = st.number_input('SkinThickness',value=35)
    with col2:
        feature5 = st.number_input('Insulin',value=0)
        feature6 = st.number_input('BMI',value=33.6)
        feature7 = st.number_input('DiabetesPedigreeFunction',value=0.627)
        feature8 = st.number_input('Age',value=50)


    # Create a DataFrame or array from inputs, matching the model's expected format
    user_input = pd.DataFrame({"Pregnancies": [feature1], "Glucose": [feature2], "BloodPressure": [feature3], "SkinThickness": [feature4], "Insulin": [feature5], "BMI": [feature6], "DiabetesPedigreeFunction": [feature7], "Age":[feature8]})

    # Make prediction
    if st.button("Predict"):
        prediction = loaded_model.predict(user_input)
        if prediction[0] == 1 :
            result="Diabetic"
        else:
            result="Non Diabetic"

        st.success(f"The predicted output is: {result}")

if __name__ == '__main__':
    main()

