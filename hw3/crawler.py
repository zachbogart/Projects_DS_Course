# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 19:28:43 2017

@author: pathouli
Modified by Zach Bogart
"""

import google
import ssl
import urllib2
import urllib
from bs4 import BeautifulSoup
import os 

def googleCrawl(resultContainer, topics):
 
    # Debug saves failure file + descriptors while running
    debug = False
    
    # Make folder for all topics
    head = resultContainer
    if not os.path.exists(head):
            os.makedirs(head)
    
    if debug: 
        failures = open("failures.txt", "w")
        failures.close()
    
    for topic in topics:
        
        print "Getting google searches on " + topic + "..."
        
        # add the directories if they aren't there
        if not os.path.exists(head + '/' + topic):
            os.makedirs(head + '/' + topic)
    
        fileIndex = list()
        
        theQuery = topic
        for url in google.search(theQuery, num=200, start=0, stop=40):
            fileIndex.append(url)
            
        cnt = 0
        for theUrl in fileIndex:
            try:
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                opener = urllib2.build_opener(urllib2.HTTPSHandler(context=ctx))
                opener.addheaders = [('Referer', theUrl)]
                html = opener.open(theUrl,timeout=10).read()
                soup = BeautifulSoup(html,"lxml")
                
                textTemp = list()
                try:
                    textTemp.append(soup.find('title').text)
                    textTemp.append('\n')
                    for theText in soup.find_all('p'): #,'li']):#,'li']):#,'ul']):#,'span']):#,'li']):
                        textTemp.append(theText.text)
                except:
                    print theUrl
                    if debug: 
                        failures = open("failures.txt", "a+")
                        failures.write(("Finding html in " + topic + " " + str(cnt) + "/" + str(len(fileIndex)) + ": " + theUrl + "\n").encode('utf8'))
                        failures.close()
                    pass    
            
                text = " " . join(textTemp)
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)
                
                # if the text is big enough, save it as a file
                if len(text.split(" ")) >= 50:
                    tmpFile = str(cnt) + ".txt"
                    indexFile = open(head + "/" + topic + "/" + tmpFile, "w")
                    indexFile.write(text.encode('utf8'))
                    indexFile.close()
                    cnt = cnt + 1
                else:
                    if debug: 
                        failures = open("failures.txt", "a+")
                        failures.write(("Too short in " + topic + " " + str(cnt) + "/" + str(len(fileIndex)) + ": " + theUrl + "\n").encode('utf8'))
                        failures.close()
                
            except:
                if debug: 
                    failures = open("failures.txt", "a+")
                    failures.write(("html setup in " + topic + " " + str(cnt) + "/" + str(len(fileIndex)) + ": " + theUrl + "\n").encode('utf8'))
                    failures.close()
                pass
            
         
        print "Finished " + topic + ":"
        print "\tNumber of files in folder: " + str(cnt) + "\n"
        