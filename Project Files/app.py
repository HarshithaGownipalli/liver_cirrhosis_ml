import os
import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Absolute path to current directory
basedir = os.path.abspath(os.path.dirname(__file__))

# Load scaler and model
with open(os.path.join(basedir, 'sc.pkl'), 'rb') as f:
    sc = pickle.load(f)

with open(os.path.join(basedir, 'model.pkl'), 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    inputs = [float(x) for x in request.form.values()]
    inputs = np.array([inputs])
    inputs = sc.transform(inputs)
    output = model.predict(inputs)

    prediction = 0 if output < 0.5 else 1
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
