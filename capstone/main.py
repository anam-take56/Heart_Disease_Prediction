from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('mode.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the user inputs from the form
        age = float(request.form['age'])
        high_bp = float(request.form['high_bp'])
        phys_hlth = float(request.form['phys_hlth'])
        high_chol = float(request.form['high_chol'])
        diabetes = float(request.form['diabetes'])
        smoker = float(request.form['smoker'])
        sex = float(request.form['sex'])
        ment_hlth = float(request.form['ment_hlth'])
        bmi = float(request.form['bmi'])

        # Create a DataFrame for the input data
        input_data = pd.DataFrame({
            'Age': [age],
            'HighBP': [high_bp],
            'PhysHlth': [phys_hlth],
            'HighChol': [high_chol],
            'Diabetes': [diabetes],
            'Smoker': [smoker],
            'Sex': [sex],
            'MentHlth': [ment_hlth],
            'BMI': [bmi]
        })

        # Make prediction
        prediction = model.predict(input_data)

        # Display the prediction (example)
        if prediction[0] == 1:
            result = 'High Risk of Heart Disease or Attack'
        else:
            result = 'Low Risk of Heart Disease or Attack'

        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
