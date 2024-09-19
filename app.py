import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

model = pickle.load(open('Anxiety-classification-model/Tranquilo-Ml-Model/model.pkl', 'rb'))

label_encoder_gender = LabelEncoder()
label_encoder_who_bmi = LabelEncoder()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    gender = request.form['gender']
    bmi = float(request.form['bmi'])
    who_bmi = request.form['who_bmi']
    depressiveness = request.form['depressiveness'] == 'True'
    depression_diagnosis = request.form['depression_diagnosis'] == 'True'
    depression_treatment = request.form['depression_treatment'] == 'True'
    anxiousness = request.form['anxiousness'] == 'True'
    anxiety_diagnosis = request.form['anxiety_diagnosis'] == 'True'
    anxiety_treatment = request.form['anxiety_treatment'] == 'True'
    sleepiness = request.form['sleepiness'] == 'True'

    gender_encoded = label_encoder_gender.fit_transform([gender])[0]
    who_bmi_encoded = label_encoder_who_bmi.fit_transform([who_bmi])[0]

    final_features = np.array([age, gender_encoded, bmi, who_bmi_encoded, depressiveness, depression_diagnosis, 
                                depression_treatment, anxiousness, 
                                anxiety_diagnosis, anxiety_treatment, 
                                sleepiness]).reshape(1, -1)


    prediction = model.predict(final_features)


    if prediction[0] == 0:
        anxiety_level = " Mild "
    elif prediction[0] == 1:
        anxiety_level = " Moderate "
    elif prediction[0] == 2:
        anxiety_level = " None-minimal "
    else:
        anxiety_level = " Severe "

    return render_template('index.html', classification_text=f"Your anxiety level: {anxiety_level}")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)