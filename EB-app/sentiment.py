import config
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import boto3

session = boto3.session.Session(region_name=config.region)
s3client = session.client('s3')
response = s3client.get_object(Bucket=config.bucket, Key=config.mpqa_path)
body_string = response['Body'].read()
mpqa = pickle.loads(body_string,encoding='latin1')

new_words = mpqa.groupby('word_').c_pol.mean().reset_index()
nw={}
for val in new_words.iterrows():
    nw[val[1][0]]=val[1][1]

analyser = SentimentIntensityAnalyzer()
analyser.lexicon.update(nw)

def sent_to_frame(df):
    neg=[]
    neu=[]
    pos=[]
    comp=[]
    for k in list(df.content):
        snt = analyser.polarity_scores(k)
        neg.append(snt['neg'])
        neu.append(snt['neu'])
        pos.append(snt['pos'])
        comp.append(snt['compound'])
    df_sub=pd.DataFrame({'neg':neg,'neu':neu,'pos':pos,'comp':comp})
    cols=df.columns.append(df_sub.columns)
    df_returned = pd.DataFrame(np.hstack([df,df_sub]),columns=cols)
    df_returned['subjectivity'] = df_returned['neu'].apply(lambda x: 1-x)
    for col in ['neg','neu','pos','comp']:
        df_returned[col]=df_returned[col].astype(float)
    return df_returned

