# Title
Analysis of semantic bias and sentiment across major American news outlets.

# Problem
The core motivation for this project is to address the increasing divergence in presentation of the same events and issues in the media. Empirically, we see a large spectrum of “spin” applied to news articles, which varies from slight bias in word choice to completely misrepresenting facts, depending on the news outlet and topic. Even if the average individual gathers their news from multiple sources, it can be challenging to discern which, if any, source is representing it accurately. In cases where individuals draw from only one source, it’s critical to understand its biases. Using Natural Language Processing, my goal is to provide average folks a tool to analyze news articles as they consume them and raise awareness of “spin” in the article.

# Methodology
The training data I’ve selected (https://www.kaggle.com/snapcrack/all-the-news) is composed of ~140k articles, from 15 major U.S. news sources, with articles dating roughly from 2011 to 2017. The initial data format is as simple as possible, mimicking what you could quickly scrape from a news site, and includes basic columns like title, publication, author, date, and article text. This data is not tagged in the supervised learning sense, i.e. it is not known from the dataset the level of bias in a given article.

The general methodology used the nltk library to process the text data to 1) tokenize/lemmatize for better matching against existing corpora 2) remove stop words to reduce noise and 3) tf-idf vectorize article text for clustering similar articles together based on topic (topic modeling).
In order to provide bias analysis, or subjectivity analysis, as well as sentiment analysis for this unsupervised learning problem, it is necessary to use a corpus or dictionary already tagged with polarity scores. I used the vaderSentiment library (https://github.com/cjhutto/vaderSentiment), which is a sentiment analysis model trained on a variety of corpora, including 500 New York Times articles, movie reviews, and Tweets. Additionally, I updated the corpora of VaderSentiment model with <a href="http://mpqa.cs.pitt.edu/corpora/mpqa_corpus/">MPQA Opinion corpus</a>, which is human-annotated corpus of ~1500 news articles and related content, to improve accuracy against the training set. Once transformed, the model returned a matrix of both sentiment (pos/neg) and neutrality, which was used to derive subjectivity (bias) for each article. 

*For further detail, see ./notebooks/final_report.pdf.

# What's included?
1. EB-app
  - This includes the components to run a light Flask app to run your own article text through the trained sentiment and subjectivity models.
  - Note that the app requires reading a large KMeans model and dataset into memory, so requires an EC2 node (if you go that route) larger than the t2.micro free tier.
2. Jupyter notebook
 - This includes data analysis of the training set, methodology for processing the data, and model training for sentiment, subjectivity, and topic modelling. 

# Getting started running the app:
1. Download EB-app
2. Install requirements.txt
3. Run application.py

Post-processed data is stored on s3 <a href="https://s3.us-east-2.amazonaws.com/ian-bury-springboard/">here</a>.
