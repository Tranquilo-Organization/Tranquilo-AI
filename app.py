import numpy as np
from flask import Flask, request, jsonify
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open('model.pkl', 'rb'))

# Initialize label encoders for categorical data
label_encoder_gender = LabelEncoder()
label_encoder_who_bmi = LabelEncoder()

@app.route('/')
def home():
    return "Welcome to the Anxiety Level Classification API. Use the /predict endpoint to make predictions."

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.json

    # Extract and process data from the request
    age = int(data['age'])
    gender = data['gender']
    bmi = float(data['bmi'])
    who_bmi = data['who_bmi']
    depressiveness = data['depressiveness']
    depression_diagnosis = data['depression_diagnosis']
    depression_treatment = data['depression_treatment']
    anxiousness = data['anxiousness']
    anxiety_diagnosis = data['anxiety_diagnosis']
    anxiety_treatment = data['anxiety_treatment']
    sleepiness = data['sleepiness']

    # Encode categorical data (gender and WHO BMI category)
    gender_encoded = label_encoder_gender.fit_transform([gender])[0]
    who_bmi_encoded = label_encoder_who_bmi.fit_transform([who_bmi])[0]

    # Create the feature array for the model
    final_features = np.array([age, gender_encoded, bmi, who_bmi_encoded, 
                               depressiveness, depression_diagnosis, 
                               depression_treatment, anxiousness, 
                               anxiety_diagnosis, anxiety_treatment, 
                               sleepiness]).reshape(1, -1)

    # Make prediction using the loaded model
    prediction = model.predict(final_features)

    # Map prediction output to anxiety level
    if prediction[0] == 0:
        anxiety_level = "Mild"
    elif prediction[0] == 1:
        anxiety_level = "Moderate"
    elif prediction[0] == 2:
        anxiety_level = "None-minimal"
    else:
        anxiety_level = "Severe"
    

    # Return the prediction as a JSON response
    return jsonify({"anxiety_level": anxiety_level})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
