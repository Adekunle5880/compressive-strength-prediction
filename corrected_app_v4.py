import numpy as np
from flask import Flask, request, render_template
import pickle
from joblib import load

app = Flask(__name__)

# Load the trained model and scaler
model = load("random_forest_model_new.pkl")
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get values from the form in the correct order
    features = [
        float(request.form['age']),
        float(request.form['weight_fine_aggregate']),
        float(request.form['weight_cement']),
        float(request.form['percentage_tmd']),
        float(request.form['weight_coarse_aggregate']),
        float(request.form['weight_tmd']),
        float(request.form['compaction_count']),
        float(request.form['slump_result']),
        float(request.form['density_cube'])
    ]

    # Preprocess the data
    features_standardize = np.array(features).reshape(1, -1)
    standardized_features = scaler.transform(features_standardize)

    # Make a prediction
    prediction = model.predict(standardized_features)

    return render_template('index.html', prediction_text=f'The Compressive Strength is {prediction[0]:.2f} N/mm2')

if __name__ == "__main__":
    app.run(debug=True)
