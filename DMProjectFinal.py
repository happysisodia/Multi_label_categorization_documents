# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 17:35:52 2019

@author: fhassan
"""

import logging
import pandas as pd
import numpy as np
from numpy import random
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup

#Exploring Data
df = pd.read_csv('projectdatanew.csv')
df = df[pd.notnull(df['types'])]
print(df.head(10))
print(df['requirement'].apply(lambda x: len(x.split(' '))).sum())

#Plotting of Data
my_types = ['Design','O&M','Construction']
plt.figure(figsize=(10,4))
df.types.value_counts().plot(kind='bar');

#Looking a few requirement and type pairs
def print_plot(index):
    example = df[df.index == index][['requirement', 'types']].values[0]
    if len(example) > 0:
        print(example[0])
        print('Type:', example[1])
print_plot(10)
print_plot(30)

#Text Preprocessing
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    # HTML decoding
    text = BeautifulSoup(text, "lxml").text 
    # lowercase text
    text = text.lower()
        # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) 
    # delete symbols which are in BAD_SYMBOLS_RE from text
    text = BAD_SYMBOLS_RE.sub('', text) 
    # delete stopwords from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) 
    
    #text = [porter.stem(text) for text in word]
    return text
    
df['requirement'] = df['requirement'].apply(clean_text)
print_plot(10)

df['requirement'].apply(lambda x: len(x.split(' '))).sum()


#Train_Test_Split
X = df.requirement
y = df.types
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)


#Naive Bayes Classification
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(X_train, y_train)

#time
from sklearn.metrics import classification_report
y_pred = nb.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=my_types))


#Linear Support Vector Machine SVM Classification
from sklearn.linear_model import SGDClassifier

sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
sgd.fit(X_train, y_train)

#time

y_pred = sgd.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=my_types))


#Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5)),
               ])
logreg.fit(X_train, y_train)

#time

y_pred = logreg.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=my_types))






