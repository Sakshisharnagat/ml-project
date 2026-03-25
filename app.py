from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Home page with input form
@app.route('/')
def home():
    return '''
    <h2>ML Model Deployment</h2>
    <form action="/predict" method="post">
        Enter Value: <input type="text" name="value">
        <input type="submit" value="Predict">
    </form>
    '''

# Prediction using form
@app.route('/predict', methods=['POST'])
def predict():
    value = int(request.form['value'])
    prediction = model.predict([[value]])
    return f"Prediction: {int(prediction[0])}"

# Prediction using URL
@app.route('/predict/<int:value>')
def predict_url(value):
    prediction = model.predict([[value]])
    return jsonify({
        "input": value,
        "prediction": int(prediction[0])
    })

# Run app
if __name__ == "__main__":
    app.run()