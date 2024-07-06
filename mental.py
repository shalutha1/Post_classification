import string
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
from flask import Flask, render_template,request, redirect
from helper import preprocessing, vectorizer, get_prediction
from logger import logging

app = Flask(__name__)

# Load your trained model
model = joblib.load("D:/ML/sample pro/classification/mental helth/model.pickle")

@app.route('/')
def index():
    return render_template('mental.html')

@app.route('/predict', methods=['POST','GET'])
def my_post():
    text = request.form['text']
    logging.info(f'Text : {text}')

    preprocessed_txt = preprocessing(text)
    logging.info(f'Preprocessed Text : {preprocessed_txt}')

    vectorized_txt = vectorizer(preprocessed_txt)
    logging.info(f'Vectorized Text : {vectorized_txt}')

    prediction = get_prediction(vectorized_txt)
    logging.info(f'Prediction : {prediction}')

    
    
  #reviews.insert(0, text)
    return redirect(request.url)



if __name__ == '__main__':
    app.run(debug=True)
