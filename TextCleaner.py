import re
import spacy
import base64
import string
import numpy as np
import pandas as pd
import seaborn as sns

from functools import reduce
import matplotlib.pyplot as plt
from collections import Counter
from textblob import TextBlob, Word

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


## reading the csv file
Dataset = pd.read_csv('data.csv')

##Dataset['word_count'] = Dataset['Requirement'].apply(lambda x: len(str(x).split(" ")))
##print(Dataset[['Requirement','word_count']].head())



#############################preprocessing functions#############################

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

#########################################################################################################

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



#####################################################
Dataset=Lemmatize(Dataset)
Dataset= LowerCase(Dataset)
Dataset=RemovePunctuation(Dataset)
Dataset=RemoveStopWords(Dataset)
Dataset=RemoveFrequentWords(Dataset,10)
Dataset=RemoveUniqueWords(Dataset)
##Dataset=AddTokens(Dataset)
##Dataset=BalanceData(Dataset,650)
#####################################################

##tfidf= GetTFIDF(Dataset,5)
##bow= GetBagOfWords(Dataset,2)

#TrainSet, TestSet = train_test_split(Dataset, test_size=0.34, random_state=60)
df = Dataset
df.to_csv('dataclean.csv', encoding='utf-8', index=False)
