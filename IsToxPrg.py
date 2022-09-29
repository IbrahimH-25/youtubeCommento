
#dependancies
from pandas import read_csv
from googleapiclient.discovery import build
from numpy import array,unique
import gravityai as grav

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle as pk
from pandas import DataFrame

def mainProcess():
    API_Key = 'AIzaSyDszE0ynG7_rducSH1jbBJt3Y2Pwb6wGoY'
    vidID = "Xb2FKFL8vJ8"
    #getting data
    data = r'youtoxic_english_1000.csv'
    dataDF = read_csv(data)
    dataDF

    #making categories
    catagoryList = dataDF.columns.values.tolist()[3:]
    catagoryList


    youtube = build('youtube','v3',
                    developerKey=API_Key)
    video_response=youtube.commentThreads().list(
    part='snippet,replies',
    videoId=vidID
    ).execute()

    commentList = []
    commentListId = []
    for item in video_response['items']:
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        commentID = item['snippet']['topLevelComment']['id']
        commentList.append(comment)
        commentListId.append(commentID)

    DFTest = DataFrame(data = {'CommentId': commentListId,'VideoId': vidID,'Text': commentList})

        

    item = video_response['items'][0]
    comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
    commentID = item['snippet']['topLevelComment']['id']
    print(comment,'\n',commentID)

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

mainProcess()