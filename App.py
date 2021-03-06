# Create API of ML model using flask

'''
This code takes the JSON data while POST request an performs the prediction using loaded model and returns
the results in JSON format.
'''

# Import libraries
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import joblib
import pickle
from PredictHelper import LSTM_predict, predict, pre_processing, vectorize_lstm, vectorize_clasifer
import keras.backend.tensorflow_backend as tb
import os

app = Flask(__name__)
cors = CORS(app)

basedir = os.path.abspath(os.path.dirname(__file__))
models_dir = os.path.join(basedir, 'models')

# Model dir
lstm_dir = os.path.join(models_dir, 'LSTM_model.pkl')
tokenizer_dir = os.path.join(models_dir, 'tokenizer_LSTM.pkl')
svm_dir = os.path.join(models_dir, 'SVM_model.pkl')
nb_dir = os.path.join(models_dir, 'NB_model.pkl')
tfidf_dir = os.path.join(models_dir, 'tfidf.pkl')


# Load the model
model_LSTM = joblib.load(lstm_dir)
tokenizer_LSTM = joblib.load(tokenizer_dir)
SVM_model = joblib.load(svm_dir)
NB_model = joblib.load(nb_dir)
tfidf_vec = joblib.load(tfidf_dir)


@app.route("/")
def main():
    return "Welcome!"


@app.route('/api/all', methods=['POST'])
# @cross_origin()
def predictApi():

    tb._SYMBOLIC_SCOPE.value = True
    # Get the data from the POST request.
    data = request.get_json()
    print(data)
    comment = data["comment"]

    # Make prediction using model loaded from disk as per the data.

    output_LSTM = LSTM_predict(comment, model_LSTM, tokenizer_LSTM)
    output_SVM = predict(comment, SVM_model, tfidf_vec)
    output_NB = predict(comment, NB_model, tfidf_vec)

    # return result
    return jsonify({"LSTM": output_LSTM, "NB": output_NB, "SVM": output_SVM})


@app.route('/api/lstm', methods=['POST'])
# @cross_origin()
def pridictLSTM():

    tb._SYMBOLIC_SCOPE.value = True

    # Get the data from the POST request.
    data = request.get_json()
    print(data)
    comment = data["comment"]

    output = LSTM_predict(comment, model_LSTM, tokenizer_LSTM)

    standardize, tokens, w_lenmatizer = pre_processing(comment)

    convert_vector = vectorize_lstm(w_lenmatizer, tokenizer_LSTM)

    string_vector = np.array_str(convert_vector)

    data = {
        'Result': output,
        'standardize': standardize,
        'tokens': tokens,
        'lenmatizer': str(w_lenmatizer),
        'vector': string_vector
    }

    return jsonify(data)


@app.route('/api/svm', methods=['POST'])
# @cross_origin()
def pridictModelSVM():
    tb._SYMBOLIC_SCOPE.value = True

    data = request.get_json()
    print(data)
    comment = data["comment"]

    output = predict(comment, SVM_model, tfidf_vec)

    standardize, tokens, w_lenmatizer = pre_processing(comment)

    convert_vector = vectorize_clasifer(w_lenmatizer, tfidf_vec)

    string_vector = str(convert_vector)

    if output is None:
        output = "NULL"

    data = {
        'Result': output,
        'standardize': standardize,
        'tokens': tokens,
        'lenmatizer': str(w_lenmatizer),
        'vector': string_vector
    }

    return jsonify(data)


@app.route('/api/nb', methods=['POST'])
# @cross_origin()
def pridictModelNP():
    tb._SYMBOLIC_SCOPE.value = True

    data = request.get_json()
    print(data)
    comment = data["comment"]

    output = predict(comment, NB_model, tfidf_vec)

    standardize, tokens, w_lenmatizer = pre_processing(comment)

    convert_vector = vectorize_clasifer(w_lenmatizer, tfidf_vec)

    string_vector = str(convert_vector)

    if output is None:
        output = "NULL"

    data = {
        'Result': output,
        'standardize': standardize,
        'tokens': tokens,
        'lenmatizer': str(w_lenmatizer),
        'vector': string_vector
    }

    return jsonify(data)


if __name__ == '__main__':
    try:
        app.run(port=5000, threaded=False, debug=True)
    except:
        print("Server is exited unexpectedly. Please contact server admin.")
