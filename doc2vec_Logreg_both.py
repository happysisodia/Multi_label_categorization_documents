import base64
import string
import re
import spacy
import gensim
import logging
import nltk
import warnings
import pandas as pd
import numpy as np

from tqdm import tqdm
import seaborn as sns
from numpy import random
from itertools import islice
from functools import reduce
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from collections import Counter
from textblob import TextBlob, Word

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from keras import layers
from keras.layers import Dense
from keras.utils import np_utils
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn import utils
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
#################################################################################################

## hide warnings##
warnings.filterwarnings('ignore')

## reading the csv file
Dataset = pd.read_csv('data.csv')
Types_List = ['O&M','Design','Construction']



#preprocessing

#lowerCase
def LowerCase(data):
    data['Requirement'] = data['Requirement'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    return data
#remove punctuation
def RemovePunctuation (data):
    data['Requirement'] = data['Requirement'].str.replace('[^\w\s]','')
    return data
#remove stopwords
def RemoveStopWords (data):
    stop = stopwords.words('english')
    data['Requirement'] = data['Requirement'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    return data

def CorrectSpelling (data):
    data['Requirement'].apply(lambda x: str(TextBlob(x).correct()))
    return data

#remove common frequent words
def RemoveFrequentWords (data, topFrequentNumber=10):
    mask = data.Type.apply(lambda x: 'Design' in x)
    FrequentDesign= pd.Series(' '.join(data[mask]['Requirement']).split()).value_counts()[:topFrequentNumber]
    mask = data.Type.apply(lambda x: 'O&M' in x)
    FrequentOM= pd.Series(' '.join(data[mask]['Requirement']).split()).value_counts()[:topFrequentNumber]
    mask = data.Type.apply(lambda x: 'Construction' in x)
    FrequentConstruction= pd.Series(' '.join(data[mask]['Requirement']).split()).value_counts()[:topFrequentNumber]
    allFrequent=[list(FrequentDesign.index),FrequentOM.index.tolist(),FrequentConstruction.index.tolist()]
    commonFrequent=list(reduce(set.intersection, [set(item) for item in allFrequent ]))
    data['Requirement'] = data['Requirement'].apply(lambda x: " ".join(x for x in x.split() if x not in commonFrequent))
    return data

def RemoveUniqueWords (data):
    counts= pd.Series(' '.join(data['Requirement']).split()).value_counts()
    to_remove = counts[counts <= 1].index
    data['Requirement'] = data['Requirement'].apply(lambda x: " ".join(x for x in x.split() if x not in to_remove))
    return data

def AddTokens (data):
    data['Tokens'] = data['Requirement'].apply(lambda x: TextBlob(x).words)
    return data

def Stem(data):
    st = PorterStemmer()
    data['Requirement'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
    return data

def Lemmatize(data):
    data['Requirement'] = data['Requirement'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    return data

def BalanceData(data,numberofsamples=600):
    data=data.groupby('Type').apply(lambda x: x.sample(numberofsamples)).reset_index(drop=True)
    return data
##########

def GetTFIDF(data,maxNGramRange=1):
    tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',
    stop_words= 'english',ngram_range=(1,maxNGramRange),sublinear_tf=True)
    Dataset_vect = tfidf.fit_transform(Dataset['Requirement'])
##    idf = tfidf.idf_
##    print(dict(zip(tfidf.get_feature_names(), idf)))
    return Dataset_vect

def GetBagOfWords(data,maxNGramRange=1):
    bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,maxNGramRange),analyzer = "word")
    Dataset_bow = bow.fit_transform(data['Requirement'])
    return Dataset_bow
def RemoveNonAlphabet(data):
	data['Requirement'] = data['Requirement'].str.replace('[^0-9a-z #+_]','')
	return data

#################################
Dataset=Lemmatize(Dataset)
Dataset= LowerCase(Dataset)
Dataset=RemovePunctuation(Dataset)
Dataset=RemoveStopWords(Dataset)
Dataset=RemoveFrequentWords(Dataset,10)
Dataset=RemoveNonAlphabet(Dataset)
Dataset=RemoveUniqueWords(Dataset)
##Dataset=AddTokens(Dataset)
tfidf= GetTFIDF(Dataset,5)
##bow= GetBagOfWords(Dataset,2)
Dataset=BalanceData(Dataset,680)
#################################

import multiprocessing
cores = multiprocessing.cpu_count()
train, test = train_test_split(Dataset, test_size=0.3, random_state=42)
import nltk
from nltk.corpus import stopwords
def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

train_tagged = train.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['Requirement']), tags=[r.Type]), axis=1)
test_tagged = test.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['Requirement']), tags=[r.Type]), axis=1)

from tqdm import tqdm

model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, workers=cores)
model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])

for epoch in range(60):
    model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha
    

def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors


y_train, X_train = vec_for_learning(model_dbow, train_tagged)
y_test, X_test = vec_for_learning(model_dbow, test_tagged)
logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
from sklearn.metrics import accuracy_score, f1_score
print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))



model_dmm = Doc2Vec(dm=1, dm_mean=1, vector_size=300, window=10, negative=5, min_count=1, workers=5, alpha=0.065, min_alpha=0.065)
model_dmm.build_vocab([x for x in tqdm(train_tagged.values)])


for epoch in range(100):
    model_dmm.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    model_dmm.alpha -= 0.002
    model_dmm.min_alpha = model_dmm.alpha


y_train, X_train = vec_for_learning(model_dmm, train_tagged)
y_test, X_test = vec_for_learning(model_dmm, test_tagged)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))

model_dbow.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
model_dmm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)


from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
new_model = ConcatenatedDoc2Vec([model_dbow, model_dmm])

def get_vectors(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors



y_train, X_train = get_vectors(new_model, train_tagged)
y_test, X_test = get_vectors(new_model, test_tagged)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))

