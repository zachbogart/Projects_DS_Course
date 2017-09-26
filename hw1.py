#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 22:01:57 2017

Zach Bogart
Homework 1
Projects in DS

"""

# Run homework
def main():
    print "Question 1a:"
    question_1a()
    print "\nQuestion 1b:"
    question_1b()
    print "\nQuestion 2:"
    question_2()
    print "\nQuestion 3:"
    question_3()


# Q1a Print only words that start with "sh"
def question_1a():
    sent = ['she', 'sells', 'sea', 'shells', 'by', 'the', 'sea', 'shore']
    for word in sent:
        if word[:2] == "sh":
            print word
        
# Q1b Print only words longer than 4 characters
def question_1b():
    sent = ['she', 'sells', 'sea', 'shells', 'by', 'the', 'sea', 'shore']
    for word in sent:
        if len(word) > 4:
            print word
            
# Q2 Append words to a list
def question_2():
    input_text = "Wild brown trout are elusive"
    word_list = list()
    words = input_text.split() # default split by spaces
    for word in words:
        word_list.append(word)
        
    print word_list
    
# Q3 Word fequency list
def question_3():
    # include packages
    import os
    import re
    from nltk.corpus import stopwords
    import csv
    # set stopwords
    stopwords = set(stopwords.words('english'))
    
    # set path
    where_i_am = os.path.dirname(os.path.realpath(__file__))
    thePath = where_i_am + '/classify/'

    # gat all of the folders within classify
    folders = os.walk(thePath).next()[1]
    
    
    # going through every folder in classify
    for folder in folders: 
        # store list of the words found in every folder
        topicWords = list() 
        # go through every file in folder
        for file in os.listdir(thePath+folder): 
            if file.endswith('.txt'):
                try:
                    # open the file
                    f = open(thePath + folder + "/" + file, "r") 
                    # format text
                    lines = f.readlines()
                    lines = [text.strip() for text in lines] # strips leading spaces
                    lines = " ".join(lines) #join sentences by spaces
                    lines = re.sub(r'[^a-zA-Z]+', ' ',lines) #remove special characters and numbers
                    lines = lines.lower() # make it all lowercase so it is just the content
                    # if it's not a stopword, add to list
                    words = lines.split() 
                    for word in words:
                        if word not in stopwords:
                            topicWords.append(word)
                    # close the file
                    f.close() 
                except:
                    pass # moving on
        
        # Store results in a dictionary for each topic
        topicFrequency = {}
        for word in topicWords:
            if word in topicFrequency:
                topicFrequency[word] = topicFrequency[word] + 1
            else:
                topicFrequency[word] = 1
                
        # create csv for every folder topic from dictionary
        with open('{}.csv'.format(folder), 'w') as csv_topic:
            print "Creating {}.csv".format(folder)
            writer = csv.writer(csv_topic)
            writer.writerow(["word", "freq"])
            for word, freq in topicFrequency.items():
               writer.writerow([word, freq])   
 
    
# Run main first    
if __name__ == "__main__":
    main()    