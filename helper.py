import numpy as np
import pandas as pd
import re
import string
import pickle

from nltk.stem import PorterStemmer
ps = PorterStemmer()

# load model
with open("D:/ML/sample pro/classification/mental helth/model.pickle", 'rb') as f:
    model = pickle.load(f)

# load stopwords
with open("D:/ML/sample pro/classification/mental helth/english", 'r') as file:
    sw = file.read().splitlines()

# load tokens
vocab = pd.read_csv("D:/ML/sample pro/classification/mental helth/vocabulary.txt", header=None)
tokens = vocab[0].tolist()

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def preprocessing(text):
    data = pd.DataFrame([text], columns=['posts'])
    data["posts"] = data["posts"].apply(lambda x: " ".join(x.lower() for x in x.split()))
    data["posts"] = data['posts'].apply(lambda x: " ".join(re.sub(r'^https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE) for x in x.split()))
    data["posts"] = data["posts"].apply(remove_punctuations)
    data["posts"] = data['posts'].str.replace('\d+', '', regex=True)
    data["posts"] = data["posts"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
    data["posts"] = data["posts"].apply(lambda x: " ".join(ps.stem(x) for x in x.split()))
    return data["posts"]

def vectorizer(ds):
    vectorized_lst = []
    for sentence in ds:
        sentence_lst = np.zeros(len(tokens))
        for i in range(len(tokens)):
            if tokens[i] in sentence.split():
                sentence_lst[i] = 1  
        vectorized_lst.append(sentence_lst)
    vectorized_lst_new = np.asarray(vectorized_lst, dtype=np.float32)
    return vectorized_lst_new

def get_prediction(vectorized_text):
    prediction = model.predict(vectorized_text)
    if prediction == -2:
        return 'very negative'
    elif prediction == -1:
        return 'negative'
    elif prediction == 0:
        return 'natural'
    elif prediction == 1:
        return 'positve'
    else:
        return 'error'