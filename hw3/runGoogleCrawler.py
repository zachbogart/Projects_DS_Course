#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 20:06:14 2017

@author: bogart
"""
'''
This file runs the homework
- crawler.py: Needs
    - topics: which topics to google search
    - debug: whether to export a failures module and additional info
- train.py: Needs
    - classifierLUT.csv for models (same folder as train)
- predict.py: Needs
    - crawler and train must run first
- streamTweet.py: Needs
    - Twitter info
    - database name
'''

import crawler
import train
import streamTweet

# where to put the data results
dataFolder = "results"
topics = ["politics", "astronomy", "medical", "music", "sports"]

databaseName = 'hw3-results'
setTerms = ["potus", "moon and the sun", "pharmacy", "drake", "quarterback"]

consumerKeyAPIKey = ''
consumerSecretAPISecret = ''
accessToken = ''
accessTokenSecret = ''

# run Part A: Google crawl for data topics
crawler.googleCrawl(dataFolder, topics)

# run Part B: train data topics
train.trainData(dataFolder)

# run Part C: get tweets, classify them, and export to mongo
streamTweet.getTweets(dataFolder, databaseName, setTerms, consumerKeyAPIKey, 
                      consumerSecretAPISecret, accessToken, accessTokenSecret)
