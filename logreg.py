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
##Dataset=Lemmatize(Dataset)
##Dataset= LowerCase(Dataset)
##Dataset=RemovePunctuation(Dataset)
##Dataset=RemoveStopWords(Dataset)
##Dataset=RemoveFrequentWords(Dataset,10)
##Dataset=RemoveNonAlphabet(Dataset)
Dataset=RemoveUniqueWords(Dataset)
##Dataset=AddTokens(Dataset)
##tfidf= GetTFIDF(Dataset,5)
##bow= GetBagOfWords(Dataset,2)
##Dataset=BalanceData(Dataset,680)
#################################


##################################preprocessing########################################
Dataset['Requirement'].apply(lambda x: len(x.split(' '))).sum()

X = Dataset.Requirement
y = Dataset.Type
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 120)
################################## End of preprocessing########################################



##################### Logistic Regression #####################
logreg = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', LogisticRegression(n_jobs=1, C=1e5)),])
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print("##################### Logistic Regression #####################")
print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=Types_List))
print("###############################################################\n\n")
###############################################################