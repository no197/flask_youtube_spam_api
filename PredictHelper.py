from flask import Flask, jsonify, request
import numpy as np
from string import digits, punctuation
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from keras.preprocessing.sequence import pad_sequences
import nltk


# Lemmatization


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def word_lenmatizer(text):
    cleanr = re.compile('<.*?>')
    text = re.sub(cleanr, ' ', text)
    clean = text.translate(str.maketrans('', '', punctuation))
    clean = clean.translate(str.maketrans('', '', digits))
    text_tokenizer = clean.split()
    # Init Lemmatizer
    lemmatizer = WordNetLemmatizer()
    hl_lemmatized = []
    lemm = [lemmatizer.lemmatize(w, get_wordnet_pos(w))
            for w in text_tokenizer]
    hl_lemmatized.append(lemm)
    return hl_lemmatized


def LSTM_predict(text, model, tokenizer):
    w_lenmatizer = word_lenmatizer(text)
    max_token = 14
    sequences = tokenizer.texts_to_sequences(w_lenmatizer)
    X = pad_sequences(sequences, maxlen=max_token)
    print(model.predict(X))
    if np.around(model.predict(X)[0]) == 1:
        print(text)
        print("=============> SPAM\n")
        return "SPAM COMMENT"
    else:
        print(text)
        print("=============> HAM\n")
    return "HAM COMMENT"


def predict(text, model, tfidf_vector):
    w_lenmatizer = word_lenmatizer(text)
    w_lenmatizer = [" ".join(x) for x in w_lenmatizer]
    X_Tfidf = tfidf_vector.transform(w_lenmatizer)
    if model.predict(X_Tfidf)[0] == 1:
        print(text)
        print("=============> SPAM\n")
        return "SPAM COMMENT"
    else:
        print(text)
        print("=============> HAM\n")
        return "HAM COMMENT"
