from flask import Flask, jsonify, request
import numpy as np
from string import digits, punctuation
import re
import string
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
import nltk


def standardize_data(text):
    # Replace email addresses with 'email'
    re_email = re.compile('[\w\.-]+@[\w\.-]+(\.[\w]+)+')
    text = re.sub(re_email, 'email', text)

    # Replace URLs with 'webaddress'
    re_url = re.compile(
        '(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?')
    text = re.sub(re_url, 'webaddress', text)

    # Replace money symbols with 'moneysymb'
    re_moneysb = re.compile('\$')
    text = re.sub(re_moneysb, 'moneysb', text)

    # Remove ufeff
    re_moneysb = re.compile('\ufeff|\\ufeff')
    text = re.sub(re_moneysb, ' ', text)

    # Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'
    re_phonenb = re.compile(
        '(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
    text = re.sub(re_phonenb, 'phonenb', text)

    # Replace numbers with 'numbr'
    re_number = re.compile('\d+(\.\d+)?')
    text = re.sub(re_number, ' numbr ', text)

    # Remove puntuation
    text = text.translate(str.maketrans('', '', punctuation))

    # Replace whitespace between terms with a single space
    re_space = re.compile('\s+')
    text = re.sub(re_space, ' ', text)

    # Remove leading and trailing whitespace
    re_space = re.compile('^\s+|\s+?$')
    text = re.sub(re_space, ' ', text)

    return text


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    token = [term for term in text.split() if term not in stop_words]
    return token


def word_lenmatizer(token):
    # Init Lemmatizer
    lemmatizer = WordNetLemmatizer()
    hl_lemmatized = []
    lemm = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in token]
    hl_lemmatized.append(lemm)
    return hl_lemmatized


def vectorize_lstm(w_lenmatizer, tokenizer):
    sequences = tokenizer.texts_to_sequences(w_lenmatizer)
    X = pad_sequences(sequences, maxlen=103)
    return X


def vectorize_clasifer(w_lenmatizer, tfidf_vector):
    w_lenmatizer = [" ".join(x) for x in w_lenmatizer]
    return tfidf_vector.transform(w_lenmatizer)


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def pre_processing(text):
    text = text.lower()
    standardize = standardize_data(text)

    tokens = remove_stopwords(standardize)

    w_lenmatizer = word_lenmatizer(tokens)

    return str(standardize), str(tokens), w_lenmatizer


def LSTM_predict(text, model, tokenizer):
    text = standardize_data(text)
    token = remove_stopwords(text)
    w_lenmatizer = word_lenmatizer(token)
    max_token = 103
    sequences = tokenizer.texts_to_sequences(w_lenmatizer)
    X = pad_sequences(sequences, maxlen=max_token)
    if np.around(model.predict(X)[0]) == 1:
        print("=============> SPAM\n")
        return "SPAM"
    else:
        print("=============> HAM\n")
        return "HAM"


def predict(text, model, tfidf_vector):
    print(text)
    text = standardize_data(text)
    token = remove_stopwords(text)
    w_lenmatizer = word_lenmatizer(token)
    w_lenmatizer = [" ".join(x) for x in w_lenmatizer]
    X_Tfidf = tfidf_vector.transform(w_lenmatizer)
    if model.predict(X_Tfidf)[0] == 1:
        print("=============> SPAM\n")
        return "SPAM"
    else:
        print("=============> HAM\n")
        return "HAM"
