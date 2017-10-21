import tweepy  
from pymongo import MongoClient
from textwrap import TextWrapper
from tweepy.utils import import_simplejson
json = import_simplejson()

import predict

auth1 = tweepy.auth.OAuthHandler('U8svDinW92huSMQNfxKQwJMFK','nDZUrP2SCL6kqLCdcSqbN45ReRajnPan2BOkrZfdRMV31Fl3Ao')  
auth1.set_access_token('4661808615-wM3V52jYLsEuRlDnsVUC8vkYjkLzNAb2pLSDuCy','6ZWesNDoFCCFqL9FlBtxlE0iL9CfLi7psajgYz5ojOjQS')  
api = tweepy.API(auth1)

mongo = MongoClient('localhost', 27017)
mongo_db = mongo['twitterDBs']
mongo_collection = mongo_db['theData']

class StreamListener(tweepy.StreamListener):  
    status_wrapper = TextWrapper(width=140, initial_indent='', subsequent_indent='')    
    def on_status(self, status): 
        tempA = self.status_wrapper.fill(status.text)
        tempB = status.retweeted 
        tempC = status.user.lang 
        tempD = status.geo
        print tempD
        if ((("en" in tempC) and (tempB is False)) and (not("RT") in tempA[:2]) and (((("http" or "www") in tempA) and ((' ') in tempA)) or (not("http" or "www") in tempA))):
            try:   
                tweetText = self.status_wrapper.fill(status.text)
                print(tweetText)
                mongo_collection.insert_one({
                'message_id': status.id,
                'screen_name': status.author.screen_name,
                'body': tweetText,
                'created_at': status.created_at,
                'followers': status.user.followers_count,
                'friends_count': status.user.friends_count,
                'location': status.user.location,
                'topic' : predict.predictText(tweetText)
                })
            except Exception, (e):  
                print("HERE")          
                pass 

l = StreamListener()  
streamer = tweepy.Stream(auth=auth1, listener=l, timeout=3000)   
setTerms = ["fishing","hiking","machine learning","mathematics"]
streamer.filter(None,setTerms)   
