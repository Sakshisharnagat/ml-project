from flask import Flask, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return "ML Model Deployment Working"

@app.route('/predict/<int:value>')
def predict(value):
    prediction = model.predict([[value]])
    return jsonify({
        "prediction": int(prediction[0])
    })

if __name__ == "__main__":
    app.run()