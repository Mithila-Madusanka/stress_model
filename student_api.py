import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the student model
clf_student = joblib.load('student.joblib')

# Load the employee model
clf_employee = joblib.load('employee.joblib')

# Define API endpoint for student predictions
@app.route('/predict/student', methods=['POST'])
def predict_student():
    # Get input data from the request
    input_data = request.json.get('input_data', [])
    input_data = [input_data]  # Assuming input_data is a list

    # Make predictions
    predictions = clf_student.predict(input_data)
    prediction = np.int64(predictions[0])
    prediction_as_int = int(prediction)

    # Return predictions as JSON response
    return jsonify({'predicted_stress_level': prediction_as_int})

# Define API endpoint for employee predictions
@app.route('/predict/employee', methods=['POST'])
def predict_employee():
    # Get input data from the request
    input_data = request.json.get('input_data', [])
    input_data = [input_data]  # Assuming input_data is a list

    # Make predictions
    predictions = clf_employee.predict(input_data)
    prediction = np.int64(predictions[0])
    prediction_as_int = int(prediction)

    # Return predictions as JSON response
    return jsonify({'predicted_stress_level': prediction_as_int})


if __name__ == '__main__':
    app.run(debug=True)