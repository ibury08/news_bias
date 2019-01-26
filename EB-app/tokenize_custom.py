import nltk
lemmatizer = nltk.stem.WordNetLemmatizer()
def lemm_words(words_list, lemmatizer):
    return [lemmatizer.lemmatize(word) for word in words_list]
    
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    lemmas = lemm_words(tokens, lemmatizer)
    return lemmas
