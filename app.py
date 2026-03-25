from flask import Flask, request, jsonify, render_template_string
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return '''
    <h2>ML Model Deployment</h2>
    <form action="/predict" method="post">
        Enter Value: <input type="text" name="value">
        <input type="submit" value="Predict">
    </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    value = int(request.form['value'])
    prediction = model.predict([[value]])
    return f"Prediction: {int(prediction[0])}"

if __name__ == "__main__":
    app.run()