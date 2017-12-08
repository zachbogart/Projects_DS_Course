#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 14:57:35 2017

@author: bogart
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pandas.io.json import json_normalize
import json
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score

printResults = True

# grab the input training data and add age bins
labeled_training_data = pd.read_csv("user_ages_train.csv")
biggest_age = max(labeled_training_data['Age'])
ageLabels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
ageBins = [0,24,34,44,54,64,biggest_age]
labeled_training_data['Age Group'] = pd.cut(labeled_training_data['Age'], bins=ageBins, labels=ageLabels)
binValues = [0,1,2,3,4,5]
labeled_training_data['bin_values'] = pd.cut(labeled_training_data['Age'], bins=ageBins, labels=binValues)

# grab user profile and twitter data files
data_profiles = json.load(open('user_age_profiles.json'))
data_tweets = json.load(open('user_age_tweets.json'))

# make profile dataframe
profiles = pd.DataFrame.from_dict(json_normalize(data_profiles), orient='columns')


# Make the tweet_stats dataframe
tweetsDict = {}

for tweet in data_tweets:
    user_ID = tweet['user']['id']
    tweetLength = tweet['text']
    
    if user_ID in tweetsDict:
        tweetsDict[user_ID]['tweet_count'] = tweetsDict[user_ID]['tweet_count'] + 1 
        tweetsDict[user_ID]['length_all_tweets'] = tweetsDict[user_ID]['length_all_tweets'] + len(tweetLength)

    else :
        tweetsDict[user_ID] = {
                'user_ID': user_ID,
                'tweet_count': 1,
                'length_all_tweets': len(tweetLength)
                }

for user in tweetsDict:
    tweetsDict[user]['avg_tweet_length'] = tweetsDict[user]['length_all_tweets'] / tweetsDict[user]['tweet_count']
    
tweet_stats = pd.DataFrame(tweetsDict.values())

    
# merge tweet_stats and profiles dataframes
df = labeled_training_data.merge(profiles, left_on="ID", right_on="id", how='inner')
df = df.merge(tweet_stats, left_on="ID", right_on="user_ID", how="inner")
 
# extract relevant features and create simplified dataframe
features = [
    'tweet_count',
    'length_all_tweets',
    'avg_tweet_length',
    'statuses_count',
    'followers_count',
    'favourites_count',
    'friends_count',
    'ID',
    'Age',
    'bin_values',
]

df1 = df[features]

# split into train/test data
train, test = train_test_split(df1, test_size=.2, random_state=0)

# Train data based on age
classifier = RandomForestClassifier(random_state=0)
classifier.fit(train, train['Age'])

# Store predictions in dataframe
result = pd.DataFrame()
result['ID'] = test['ID']
result['age'] = test['Age']
result['age_prediction'] = classifier.predict(test)

# Add group columns for accuracy scoring
result['age_prediction_group'] = pd.cut(result['age_prediction'], bins=ageBins, labels=ageLabels)
result['actual_age_group'] = pd.cut(result['age'], bins=ageBins, labels=ageLabels)

# print accuracy results
print "Group Accuracy: " + str(accuracy_score(result['actual_age_group'], result['age_prediction_group']))
print "Age Accuracy: " + str(accuracy_score(result['age'], result['age_prediction']))


# export to CSV, if you want to
if printResults:
    printable = result[['ID', 'age_prediction_group']].copy()
    
    printable.to_csv('age_predictions.csv')












