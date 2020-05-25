# Create API of ML model using flask

'''
This code takes the JSON data while POST request an performs the prediction using loaded model and returns
the results in JSON format.
'''

# Import libraries
import numpy as np
from flask import Flask, request, jsonify
import joblib
import pickle
from PredictHelper import LSTM_predict, predict

app = Flask(__name__)

# Load the model
model_LSTM = joblib.load('LSTM_model.pkl')
tokenizer_LSTM = joblib.load('tokenizer_LSTM.pkl')
SVM_model = joblib.load('SVM_model.pkl')
NB_model = joblib.load('NB_model.pkl')
tfidf_vec = joblib.load('tfidf.pkl')


@app.route("/")
def main():
    return "Welcome!"


@app.route('/api', methods=['POST'])
def predictApi():
    # Get the data from the POST request.
    data = request.get_json()
    print(data)
    comment = data["comment"]

    # Make prediction using model loaded from disk as per the data.

    output_LSTM = LSTM_predict(comment, model_LSTM, tokenizer_LSTM)

    output_NB = predict(comment, NB_model, tfidf_vec)

    # return result
    return jsonify({"LSTM": output_LSTM, "NB": output_NB, })


if __name__ == '__main__':
    try:
        app.run(port=5000, threaded=False)
    except:
        print("Server is exited unexpectedly. Please contact server admin.")
