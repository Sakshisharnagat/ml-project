from flask import Flask, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Home route
@app.route('/')
def home():
    return "ML Model Deployment Working"

# Prediction route
@app.route('/predict/<int:value>')
def predict(value):
    prediction = model.predict([[value]])
    return jsonify({
        "input": value,
        "prediction": int(prediction[0])
    })

# Run app
if __name__ == "__main__":
    app.run()