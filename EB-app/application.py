from flask import Flask, render_template, flash, redirect,session, Response
from flask_restful import reqparse, abort, Api, Resource
import pickle
import re
import numpy as np
import nltk
import os
from tokenize_custom import lemm_words, tokenize
import pandas as pd
from flask_wtf import FlaskForm
from wtforms import TextAreaField
from wtforms.validators import DataRequired
import config
import sentiment
import matplotlib as mpl
mpl.use('agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
import matplotlib.pyplot as plt
import boto3
import sys
from urllib.request import urlopen
from io import BytesIO
import base64
import seaborn as sns


application = Flask(__name__)
application.config.from_object(config.Config)
api = Api(application)


def try_to_load_as_pickled_object(filepath):
    """
    This is a defensive way to write pickle.load, allowing for very large files on all platforms
    """
    max_bytes = 2**31 - 1
    input_size = os.path.getsize(filepath)
    bytes_in = bytearray(0)
    with open(filepath, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    obj = pickle.loads(bytes_in)
    return obj


class MyCustomUnpickler(pickle.Unpickler):
    """
    Required unpickler to all unpickling of vectorizer using custom tokenizer function tokenize_custom.py
    """
    def find_class(self, module, name):
        if module == "__main__":
            module = "tokenize_custom"
        return super().find_class(module, name)


vectorizer_path = config.vectorizer_path

session = boto3.session.Session(region_name=config.region)
s3client = session.client('s3')
print('loading resources')
response = s3client.get_object(Bucket=config.bucket, Key=config.data_path)
body_string = response['Body'].read()
data = pickle.loads(body_string)
print("data loaded")
response = s3client.get_object(Bucket=config.bucket, Key=config.model_path)
body_string = response['Body'].read()
model = pickle.loads(body_string)
print("model loaded")


with urlopen(vectorizer_path) as f:
    unpickler = MyCustomUnpickler(f)
    vectorizer = unpickler.load()
    
print('vectorizer loaded')



    
# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')

download_dir='/opt/python/current/app'
nltk.data.path.append(download_dir)
try:
    nltk.download('stopwords',download_dir=download_dir)
    nltk.download('punkt',download_dir=download_dir)
    nltk.download('wordnet',download_dir=download_dir)
except:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

def clean(raw):
    """
    Function to clean text to keep only letters and remove stopwords
    Returns a string of the cleaned raw text
    """
    letters_only = re.sub('[^a-zA-Z]', ' ', raw)
    words = letters_only.lower().split()
    stopwords_eng = set(nltk.corpus.stopwords.words("english"))
    useful_words = [x for x in words if not x in stopwords_eng]
    
    # Combine words into a paragraph again
    useful_words_string = ' '.join(useful_words)
    return(useful_words_string)

def process_request(user_query:str):
    """
    Function to clean, vectorize, and get topic_id for inputted text. 
    Also returns most subjectivy articles within the same topic.
    """
    cleaned = clean(user_query)
    v = vectorizer.transform([cleaned])
    topic = model.predict(v)
    topic_id = int(topic[0])
    related_articles = data[(data.topic_id==topic_id)&(data.subjectivity_content>0)&(data.has_url)].sort_values('subjectivity_content',ascending=True).head().to_dict('records')
    output = {'topic_id': topic_id,'related_articles': related_articles}
    return output



class QueryForm(FlaskForm):
    query = TextAreaField('Article Content:', validators=[DataRequired()],render_kw={"rows": 30, "cols": 120})


def create_figure(value):
    fig = plt.figure(figsize=(8, 2))
    ax = fig.add_axes([0.1, 0.4, 0.8, 0.2])

    bounds = [-1, -0.5, 0, 0.5, 1]
    labels = ('very negative', 'somewhat negative', 'somewhat positive', 'very positive')

    cmap = mpl.cm.coolwarm
    norm = mpl.colors.Normalize(vmin=bounds[0], vmax=bounds[-1])

    cb = mpl.colorbar.ColorbarBase(
        ax,
        cmap=cmap,
        norm=norm,
        orientation='horizontal',
    
        label='Article Sentiment Score',
        )

    for i, label in enumerate(labels):
        xpos = float((2*i + 1))/(2*len(labels))
        ax.annotate(label, xy=(xpos, 0.5), xycoords='axes fraction', ha='center', va='center')
    cb.ax.plot(value,  0, 'k.', markersize=30)

    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
    
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png




def create_figure2(value):
    fig2 = plt.figure(figsize=(8, 2))
    ax = fig2.add_axes([0.1, 0.4, 0.8, 0.2])

    bounds = [-1, -0.5, 0, 0.5, 1]
    labels = ('very objective', 'somewhat objective', 'somewhat subjective', 'very subjective')

    cmap = mpl.cm.coolwarm
    norm = mpl.colors.Normalize(vmin=bounds[0], vmax=bounds[-1])

    cb = mpl.colorbar.ColorbarBase(
        ax,
        cmap=cmap,
        norm=norm,
        orientation='horizontal',
    
        label='Article Subjectivity Score',
        )

    for i, label in enumerate(labels):
        xpos = float((2*i + 1))/(2*len(labels))
        ax.annotate(label, xy=(xpos, 0.5), xycoords='axes fraction', ha='center', va='center')
    cb.ax.plot(value, 0, 'k.', markersize=30)
    
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file


    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png


def scatter(tid,x,y):
    data_plot = data[data.topic_id==tid].groupby('publication')[['comp','subjectivity_content']].median().reset_index().append(pd.DataFrame({'publication':'Your Article','comp':[y],
             'subjectivity_content':[x]}))

    markers = {}
    for n,a in enumerate(data_plot.publication.values):
        markers[a] = "." if n<10 else "*"
    fig=plt.figure(figsize=(6,4))
    sns.set_palette('deep',n_colors=16)
    _=sns.scatterplot(data=data_plot,x='subjectivity_content',y='comp', hue='publication',palette='deep', legend='full', s=200, style='publication',markers=markers)
    ax=plt.gca()
    text=ax.annotate('Your Article',(x,y))
    lgd=ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    _=plt.xlabel('Topic Subjectivity')
    _=plt.ylabel('Topic Sentiment')
    _=plt.title('Median Sentiment, Subjectivity by Source on this Topic')
    figfile = BytesIO()
    plt.savefig(figfile, format='png', bbox_extra_artists=(lgd,text), bbox_inches='tight')
    figfile.seek(0)  # rewind to beginning of file


    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png





@application.route('/',methods=['GET','POST'])
def main():
    form = QueryForm()
    if form.validate_on_submit():
        query = form.query.data
        #Finding related articles
        output_dict = process_request(query)
        topic_id = output_dict['topic_id']
        articles = output_dict['related_articles']
        #Sentiment and subjectivity for current article
        input_article = pd.DataFrame({'content':[query]})
        res = sentiment.sent_to_frame(input_article)
        sentiment_score = round(res['comp'][0],3)
        subjectivity = round(res['subjectivity'][0],3)

        subj = create_figure2(subjectivity)
        sent = create_figure(sentiment_score)
        scatter_plot = scatter(topic_id,subjectivity,sentiment_score)


        
        return render_template('results.html', topic_id=topic_id,articles=articles, sentiment_score=sentiment_score, subjectivity=subjectivity, subj=subj.decode('utf8'), sent=sent.decode('utf8'),scatter=scatter_plot.decode('utf8'))
    return render_template('home.html',form=form)



    





if __name__ == '__main__':
    application.run(debug=True)
    application.secret_key = 'you-will-never-guess'
