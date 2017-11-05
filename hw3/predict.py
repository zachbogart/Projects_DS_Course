# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.externals import joblib
import sys
import nltk
from nltk.corpus import stopwords
import re
import os

def genCorpus(theText):
#    setTerms = set(setTerms)
    
    #set dictionaries
    stopWords = set(stopwords.words('english'))
    theStemmer = nltk.stem.porter.PorterStemmer() #Martin Porters celebrated stemming algorithm
    
    #pre-processing
    theText = theText.split()
    tokens = [token.lower() for token in theText] #ensure everything is lower case
    tokens = [re.sub(r'[^a-zA-Z0-9]+', ' ',token) for token in tokens] #remove special characters but leave word in tact
    tokens = [token for token in tokens if token.lower().isalpha()] #ensure everything is a letter
    tokens = [word for word in tokens if word not in stopWords] #rid of stop words    
    tokens = [theStemmer.stem(word) for word in tokens] #stem words uing porter stemming algorithm
    tokens = " ".join(tokens) #need to pass string seperated by spaces       
    print tokens
    return tokens

    #function of whole file
def predictText(tempText, resultContainer):
    # where the classify files are
    thePath = os.path.dirname(os.path.realpath(__file__)) + '/' + resultContainer + '/'
    theCols = os.walk(thePath).next()[1] 
    path = os.path.dirname(os.path.realpath(__file__)) + '/'
    # where all the files are that train.py generated
    vectorizer = joblib.load(path + 'vectorizer.pk') 
    pca = joblib.load(path + 'pca.pk') 
    for file in os.listdir(path):
        if file.endswith(".pkl"):
            theFile = file
    model = joblib.load(path + theFile) #manual for now      
    testText = list()
    testText.append(genCorpus(tempText))
    test = vectorizer.transform(testText)
    X2_new = pca.transform(test.toarray())
    x = model.predict(X2_new)
    x = x[0]
    xProba = pd.DataFrame(model.predict_proba(X2_new))
#    print xProba
    xProba = xProba.round(4) 
#    print xProba
#    print xProba.columns
#    xProba.columns=theCols
    #xProba = xProba.to_json()
#    sys.stdout.write(xProba)
    #sys.stdout.write('\n')
#    print xProba
    return theCols[xProba.columns[x]]
    
#setTerms = ["potus", "moon and the sun", "pharmacy", "drake", "quarterback"]
#print genCorpus("bears are drake potus pharmacy when they go fishing for trout", setTerms)
#print predictText("lkjhlkjhs", "results", setTerms)
    
    #echo -e '{"path": "C:/rawData/motivations/","text": "this is a test of broadcast system"}' | python predict.py