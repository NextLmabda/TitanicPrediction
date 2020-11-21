import numpy as np
import pandas as pd
import pickle

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods = ['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    value = model.predict([features])
    #features = [np.array(features)]
    #print(model.predict([features]))
    if value[0] == 0:
        return render_template('index.html', prediction_text = 'The person {}'.format('Survived'))
    else:
        return render_template('index.html', prediction_text='The person {}'.format('Did not survive'))

if __name__ == '__main__':
    app.run(debug = True)

