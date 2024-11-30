import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained model
model = pickle.load(open("model.pkl", "rb"))

# Initialize label encoders for categorical data
label_encoder_gender = LabelEncoder()
label_encoder_who_bmi = LabelEncoder()

# Anxiety level mapping
anxiety_levels = [
    {"level": "None-minimal", "id": 4},
    {"level": "Mild", "id": 1},
    {"level": "Moderate", "id": 2},
    {"level": "Severe", "id": 3},
]

# Streamlit app setup
st.title("Anxiety Level Classification")
st.write("Use this app to classify anxiety levels based on your input data.")

# Input fields
st.sidebar.header("Input Data")
age = st.sidebar.number_input("Age", min_value=0, step=1, value=25)
gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
bmi = st.sidebar.number_input("BMI", min_value=0.0, step=0.1, value=22.5)
who_bmi = st.sidebar.selectbox(
    "WHO BMI Category", options=["Underweight", "Normal", "Overweight", "Obese"]
)
depressiveness = st.sidebar.slider(
    "Depressiveness (0-10)", min_value=0, max_value=10, value=5
)
depression_diagnosis = st.sidebar.selectbox(
    "Previous Depression Diagnosis", options=[0, 1]
)
depression_treatment = st.sidebar.selectbox(
    "Currently Receiving Depression Treatment", options=[0, 1]
)
anxiousness = st.sidebar.slider(
    "Anxiousness (0-10)", min_value=0, max_value=10, value=5
)
anxiety_diagnosis = st.sidebar.selectbox("Previous Anxiety Diagnosis", options=[0, 1])
anxiety_treatment = st.sidebar.selectbox(
    "Currently Receiving Anxiety Treatment", options=[0, 1]
)
sleepiness = st.sidebar.slider("Sleepiness (0-10)", min_value=0, max_value=10, value=5)

# Button to make a prediction
if st.sidebar.button("Classify Anxiety Level"):
    try:
        # Encode categorical data
        gender_encoded = label_encoder_gender.fit_transform([gender])[0]
        who_bmi_encoded = label_encoder_who_bmi.fit_transform([who_bmi])[0]

        # Create the feature array for the model
        final_features = np.array(
            [
                age,
                gender_encoded,
                bmi,
                who_bmi_encoded,
                depressiveness,
                depression_diagnosis,
                depression_treatment,
                anxiousness,
                anxiety_diagnosis,
                anxiety_treatment,
                sleepiness,
            ]
        ).reshape(1, -1)

        # Make prediction using the loaded model
        prediction = model.predict(final_features)

        # Get the predicted anxiety level and ID
        predicted_anxiety = (
            anxiety_levels[prediction[0]]
            if prediction[0] < len(anxiety_levels)
            else {"level": "Unknown", "id": -1}
        )

        # Display the prediction
        st.success(f"Predicted Anxiety Level: {predicted_anxiety['level']}")
        st.info(f"Anxiety Level ID: {predicted_anxiety['id']}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
