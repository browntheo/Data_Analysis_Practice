#https://github.com/WuraolaOyewusi/Opinion-Mining-using-the-UCI-Drug-Review-Dataset/blob/master/Data_Loading_and_preprocessing(drug_Review_Dataset).ipynb
import re
import nltk
import pandas as pd
df1 = pd.read_csv('drugsComTest_raw.tsv',delimiter='\t')     # Read the files with the pandas dataFrame
df2 = pd.read_csv('drugsComTrain_raw.tsv', delimiter='\t')   #  pass use the '\t' delimiter as argument because it is a tab separated file to prevent parser error
df = pd.concat([df1,df2])  # combine the two dataFrames into one for a bigger data size and ease of preprocessing
df1.shape
df2.shape
df.shape
df.head()
df.columns = ['Id','drugName','condition','review','rating','date','usefulCount']    #rename columns
df.head
df['date'] = pd.to_datetime(df['date'])    #convert date to datetime eventhough we are not using date in this
df['date'].head()             #confirm conversion
df2 = df[['Id','review','rating']].copy()    # create a new dataframe with just review and rating for sentiment analysis
df2.head()
df2.isnull().any().any()    # check for null
df2.info(null_counts=True)         #another way to check for null
df2.info()       #check for datatype, also shows null
df2['Id'].unique()       # shows unique Id as array
df2['Id'].count()      #count total number of items in the Id column
df2['Id'].nunique()     #shows unique Id values

df['review'][1]         # access indivdual value
df.review[1]            # another method to assess individual value in a Series
#!pip install vaderSentiment       #install Sentiment Analysis  library
import nltk
nltk.download(['punkt','stopwords'])
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
df2.head()
df2['cleanReview'] = df2['review'].apply(lambda x: ' '.join([item for item in x.split() if item not in stopwords]))     # remove stopwords from review
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
df2['vaderReviewScore'] = df2['cleanReview'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
df2.head()
positive_num = len(df2[df2['vaderReviewScore'] >=0.05])
neutral_num = len(df2[(df2['vaderReviewScore'] >-0.05) & (df2['vaderReviewScore']<0.05)])
negative_num = len(df2[df2['vaderReviewScore']<=-0.05])
positive_num,neutral_num, negative_num
df2['vaderSentiment']= df2['vaderReviewScore'].map(lambda x:int(2) if x>=0.05 else int(1) if x<=-0.05 else int(0) )
df2['vaderSentiment'].value_counts()
Total_vaderSentiment = positive_num + neutral_num + negative_num
Total_vaderSentiment
df2.loc[df2['vaderReviewScore'] >=0.05,"vaderSentimentLabel"] ="positive"
df2.loc[(df2['vaderReviewScore'] >-0.05) & (df2['vaderReviewScore']<0.05),"vaderSentimentLabel"]= "neutral"
df2.loc[df2['vaderReviewScore']<=-0.05,"vaderSentimentLabel"] = "negative"
df2.shape
df2.head()
positive_rating = len(df2[df2['rating'] >=7.0])
neutral_rating = len(df2[(df2['rating'] >=4) & (df2['rating']<7)])
negative_rating = len(df2[df2['rating']<=3])
positive_rating,neutral_rating,negative_rating
Total_rating = positive_rating+neutral_rating+negative_rating
Total_rating
df2['ratingSentiment']= df2['rating'].map(lambda x:int(2) if x>=7 else int(1) if x<=3 else int(0) )
df2['ratingSentiment'].value_counts()
df2.head()
df2.loc[df2['rating'] >=7.0,"ratingSentimentLabel"] ="positive"
df2.loc[(df2['rating'] >=4.0) & (df2['rating']<7.0),"ratingSentimentLabel"]= "neutral"
df2.loc[df2['rating']<=3.0,"ratingSentimentLabel"] = "negative"
df2.head()
df2 = df2[['Id','review','cleanReview','rating','ratingSentiment','ratingSentimentLabel','vaderReviewScore','vaderSentiment','vaderSentimentLabel']]
df2.head()
df2.to_csv('processed.csv')    # To save preprocessed dataset to csv
df2.head(50)
import os
os.stat('processed.csv').st_size         # Check size of csv file About 181MB
df2.info()
df2.to_csv('processed.csv.gz',compression='gzip')
os.stat('processed.csv.gz').st_size    #compressed to about 54MB