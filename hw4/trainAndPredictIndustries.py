import pandas as pd
import numpy as np
from sklearn import feature_extraction
from scipy import stats
from sklearn import decomposition,linear_model
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier
from sklearn.manifold import Isomap
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier,Lasso,SGDClassifier,LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,f1_score,hamming_loss
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn
import pandas as pd
import numpy
import nltk
import os
import re
from nltk.corpus import stopwords,wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn import decomposition
import time
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier
from sklearn.manifold import Isomap
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import csv
import sys


# where the classifierLUT.csv file is
thePathLut = os.path.dirname(os.path.realpath(__file__)) + "/"
#    theCols = os.walk(thePath).next()[1]   

finalWords = list()
theDocs = list()

def genCorpus(theText):
    #set dictionaries
    stopWords = set(stopwords.words('english'))
    theStemmer = nltk.stem.porter.PorterStemmer() #Martin Porters celebrated stemming algorithm
#        print theText
#        print '\n'
    #pre-processing
    theText = theText.split()
    tokens = [token.lower() for token in theText] #ensure everything is lower case
#        print tokens
    tokens = [re.sub(r'[^a-zA-Z0-9]+', '',token) for token in tokens] #remove special characters but leave word in tact
#        print tokens
    tokens = [token for token in tokens if token.isalpha()] #ensure everything is a letter
#        print tokens
#        tokens = [word for word in tokens if word not in stopWords] #rid of stop words
#        tokens = [theStemmer.stem(word) for word in tokens] #stem words uing porter stemming algorithm
    tokens = " ".join(tokens) #need to pass string seperated by spaces       
    return tokens

def textToNum(theLabels,thePredLabel):
    theOutLabel = dict() 
    cnt = 0
    for word in theLabels:
        theOutLabel[word] = cnt
        cnt = cnt + 1
    return str(theOutLabel[thePredLabel])

theLUT = pd.read_csv(thePathLut + 'classifierLUT.csv',index_col=0) #ALGO LUT
def optFunc(theAlgo,theParams):
    theModel = theLUT.loc[theAlgo,'optimizedCall']
    tempParam = list()
    for key, value in theParams.iteritems():
        tempParam.append(str(key) + "=" + str(value)) 
    theParams = ",".join(tempParam)
    theModel = theModel + theParams + ")"
    return theModel 

def algoArray(theAlgo):
    theAlgoOut = theLUT.loc[theAlgo,'functionCall']
    return theAlgoOut



'''
Changes:
    - Add in csv file
    - Sample for every industry
    - Use sample to train model
'''

#read in CSV
entireFile = pd.read_csv("private us companies.csv")
del entireFile['Unnamed: 0']
del entireFile['Unnamed: 3']
#entireFile.head()

#make dict of industry counts
industries = dict()

for ind in entireFile['Industry']:
    if ind not in industries:
        industries[ind] = 1
    else:
        industries[ind] += 1
        
#create sample and over/undersample based on industry count
sampleSize = 100
sample = pd.DataFrame(columns= ['Company', 'Industry'])

for ind in industries:
    if industries[ind] < 100:
        #oversample, with replacement (replace True)
        justOneIndustry = entireFile[(entireFile.Industry == ind)]
        smallSample = justOneIndustry.sample(n=sampleSize, replace=True)
        sample = pd.concat([smallSample, sample])
    else:
        #undersample, without replacement
        justOneIndustry = entireFile[(entireFile.Industry == ind)]
        smallSample = justOneIndustry.sample(n=sampleSize, replace=False)
        sample = pd.concat([smallSample, sample])
      
#add results to train variables
for company in sample['Company']:
    finalWords.append(genCorpus(company))
for industry in sample['Industry']:
    theDocs.append(industry)


      
vectorizer = TfidfVectorizer(max_features=1000,ngram_range=(1,1))
tdm = pd.DataFrame(vectorizer.fit_transform(finalWords).toarray())

with open('vectorizer.pk', 'wb') as fin:
    pickle.dump(vectorizer, fin)

tdm.columns=vectorizer.get_feature_names()
tdm.index=theDocs

pca = decomposition.PCA(n_components=.95)
pca.fit(tdm)
reducedTDM = pd.DataFrame(pca.transform(tdm)) #reduced tdm distance matrix

with open('pca.pk', 'wb') as fin:
    pickle.dump(pca, fin)

reducedTDM.index=theDocs

pcaVar = round(sum(pca.explained_variance_ratio_),2)

fullIndex = reducedTDM.index.values
#    fullIndex = [int(word.split("_")[0]) for word in fullIndex]


theModels = ['RF']#,'ABDT']#,'LOGR']#,'NN']#,'DT','ABDT','LDA']#,'DT','LDA','BAG','KNN','NN'] #these MUST match up with names from LUT #ABDT, #GBC, #RSM take far too long
theResults = pd.DataFrame(0,index=theModels,columns=['accuracy','confidence','runtime'])
for theModel in theModels:
    startTime = time.time()
    model = eval(algoArray(theModel))
    #model = RandomForestClassifier(random_state=50)
    print(theModel)

    #cross validation    
    cvPerf = cross_val_score(model,reducedTDM,fullIndex,cv=10)
    theResults.ix[theModel,'accuracy'] = round(cvPerf.mean(),2)
    theResults.ix[theModel,'confidence'] = round(cvPerf.std() * 2,2)
    endTime = time.time()
    theResults.ix[theModel,'runtime'] = round(endTime - startTime,0)
    
print(theResults)

#############################################
#######Run with best performing model########
#####Fine Tune Algorithm Grid Search CV######
#############################################
bestPerfStats = theResults.loc[theResults['accuracy'].idxmax()]
modelChoice = theResults['accuracy'].idxmax()
              
startTime = time.time()
model = eval(algoArray(modelChoice))
grid = GridSearchCV(estimator=model, param_grid={"n_estimators": [10,30,50,100]})#eval(gridSearch(modelChoice))
grid.fit(reducedTDM,fullIndex)
#grid.fit(train,trainIndex)
bestScore = round(grid.best_score_,4)
parameters = grid.best_params_
endTime = time.time()
print("Best Score: " + str(bestScore) + " and Grid Search Time: " + str(round(endTime - startTime,0)))

############################################
######Train Best Model on Full Data Set#####
########Save Model for future use###########
############################################
startTime = time.time()
model = eval(optFunc(modelChoice,parameters)) #train fully validated and optimized model
model.fit(reducedTDM,fullIndex)
#model.fit(train,trainIndex)
#joblib.dump(model, modelChoice + '.pkl') #save model
endTime = time.time()
print("Model Fit Time: " + str(round(endTime - startTime,0)))
    

# returns prediction dataframe for all industries
def predictIndustry(tempText, model):
    
    testText = list()
    testText.append(genCorpus(tempText))
    test = vectorizer.transform(testText)
    X2_new = pca.transform(test.toarray())
    industryPrediction = pd.DataFrame(model.predict_proba(X2_new))
    industryPrediction = industryPrediction.round(4) 
    allIndustries = model.classes_.tolist()
    industryPrediction.columns=allIndustries
    industryPrediction = industryPrediction.transpose()
    industryPrediction = industryPrediction.sort_values(0, ascending=False)
    return industryPrediction

'''
Here are some examples. Feel free to run your own.
- change tempText with desired phrase
'''
tempText = "aerospace, rockets, and space shuttles"
prediction = predictIndustry(tempText, model)

print "\nInput Text: " + tempText
print "Industry Predictions:"
print prediction.head(10)

tempText = "Buy and sell stocks to make money. Finance."
prediction = predictIndustry(tempText, model)

print "\nInput Text: " + tempText
print "Industry Predictions:"
print prediction.head(10)

tempText = "Beverages are tasty. Soda and Tea."
prediction = predictIndustry(tempText, model)

print "\nInput Text: " + tempText
print "Industry Predictions:"
print prediction.head(10)

