import tweepy  
from pymongo import MongoClient
from textwrap import TextWrapper
from tweepy.utils import import_simplejson
json = import_simplejson()

import predict

#auth1 = tweepy.auth.OAuthHandler('<Consumer Key (API Key)>','<Consumer Secret (API Secret)>')  
#auth1.set_access_token('<Access Token>','<Access Token Secret>')  

def getTweets(resultContainer, databaseName, setTerms, consumerKeyAPIKey, consumerSecretAPISecret, accessToken, accessTokenSecret):
    
    auth1 = tweepy.auth.OAuthHandler(consumerKeyAPIKey,consumerSecretAPISecret)  
    auth1.set_access_token(accessToken,accessTokenSecret)  
    api = tweepy.API(auth1)
    
    mongo = MongoClient('localhost', 27017)
    mongo_db = mongo[databaseName]
    mongo_collection = mongo_db['theData']

    
    class StreamListener(tweepy.StreamListener):  
        status_wrapper = TextWrapper(width=140, initial_indent='', subsequent_indent='')    
        def on_status(self, status): 
            tempA = self.status_wrapper.fill(status.text)
            tempB = status.retweeted 
            tempC = status.user.lang 
            tempD = status.geo
#            print tempD
            if ((("en" in tempC) and (tempB is False)) and (not("RT") in tempA[:2]) and (((("http" or "www") in tempA) and ((' ') in tempA)) or (not("http" or "www") in tempA))):
                try:   
                    tweetText = self.status_wrapper.fill(status.text)
                    prediction = predict.predictText(tweetText, resultContainer)
                    print(tweetText)
                    print(prediction)
                    mongo_collection.insert_one({
                    'message_id': status.id,
                    'screen_name': status.author.screen_name,
#                    'body': tweetText,
                    'created_at': status.created_at,
                    'followers': status.user.followers_count,
                    'friends_count': status.user.friends_count,
                    'location': status.user.location,
                    'topic' : prediction
                    })
                except Exception, (e):  
                    print("HEY")
                    print(e)
                    pass 
    
    l = StreamListener()  
    streamer = tweepy.Stream(auth=auth1, listener=l, timeout=3000)   
    streamer.filter(None,setTerms)   
