
#dependancies
from os import read
from pandas import api, read_csv
from googleapiclient.discovery import build
from numpy import array,unique
import gravityai as grav

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle as pk
from pandas import DataFrame





def getAPIKeyAndVideoId():
    with open('input.txt','r') as inputFile:
        inStream = inputFile.read()
    wordStream = inStream.split(',')
    apiKey = wordStream[0]
    videoId = wordStream[1]
    return apiKey,videoId


def getAllCommentsToLists(API_Key,vidID):
    youtube = build('youtube','v3',
                    developerKey=API_Key)
    video_response_obj=youtube.commentThreads()
    
    video_response = video_response_obj.list(
    part='snippet,replies',
    videoId=vidID,
    maxResults = 100
    ).execute()

    commentList = []
    commentListId = []
    for item in video_response['items']:
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        commentID = item['snippet']['topLevelComment']['id']
        commentList.append(comment)
        commentListId.append(commentID)

    nextPageTken = video_response.get('nextPageToken')

    while nextPageTken:
        video_response = video_response_obj.list(
        part='snippet,replies',
        videoId=vidID,
        maxResults = 100,
        pageToken = nextPageTken
        ).execute()

        for item in video_response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            commentID = item['snippet']['topLevelComment']['id']
            commentList.append(comment)
            commentListId.append(commentID)
        nextPageTken = video_response.get('nextPageToken')
        print(nextPageTken)
    
    return commentList,commentListId


def mainProcess():
    API_Key = ''

    vidID = ''

    try :
        API_Key,vidID =  getAPIKeyAndVideoId()
        print(API_Key,vidID)
    except:
        print("ERROR: Cannot find api key or video id in input.txt") 
    #vidID = "Xb2FKFL8vJ8"
    #getting data

    if API_Key == '' or vidID == '':
        raise Exception("ERROR: API_Key or vidID not set")


    
    catagoryList = ['IsToxic', 'IsAbusive', 'IsThreat', 'IsProvocative', 'IsObscene', 'IsHatespeech', 'IsRacist', 'IsNationalist', 'IsSexist', 'IsHomophobic', 'IsReligiousHate', 'IsRadicalism']
    #making categories

    commentList,commentListId = getAllCommentsToLists(API_Key,vidID)
   

    DFTest = DataFrame(data = {'CommentId': commentListId,'VideoId': vidID,'Text': commentList})

        

    '''item = video_response['items'][0]
    comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
    commentID = item['snippet']['topLevelComment']['id']
    print(comment,'\n',commentID)'''

    dataTestDF = DFTest[['CommentId','VideoId','Text']]
    for cat in catagoryList:
        #creating a test data that will be the output
        
        # creating code for model


        model = pk.load(open('pklFiles//%s_classifier.pkl'%cat,'rb'))
        tdif_vectorizer = pk.load(open('pklFiles//%s_vectorized_x.pkl'%cat,'rb'))
        #lable_encoder = pk.load(open(''))

        features = tdif_vectorizer.transform(dataTestDF['Text'])
        predictions = model.predict(features)
        #print(predictions)
        dataTestDF[cat] = predictions
    print(dataTestDF)
    dataTestDF.to_csv('commentAnalysis.csv')

grav.wait_for_requests(mainProcess)