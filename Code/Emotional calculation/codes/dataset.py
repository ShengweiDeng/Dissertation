"""
@Author: Shengwei Deng
@Software: PyCharm
"""

import re
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.utils import shuffle
# remove stop words
def remove_stopwords(stopwords,word_list):
    res = []
    for item in word_list:
        if item not in stopwords:
            res.append(item)
    return res
# text filter
def character_filter(content): #
    text = content.lower()
    # removing unwanted digits ,special chracters from the text
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)", " ", text).split())  # tags
    text = ' '.join(re.sub("^@?(\w){1,15}$", " ", text).split())

    text = ' '.join(re.sub("(\w+:\/\/\S+)", " ", text).split())  # Links
    text = ' '.join(
        re.sub("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", " ", text).split())
    text = ' '.join(re.sub(r'http\S+', '', text).split())

    text = ' '.join(re.sub(r'www\S+', '', text).split())
    text = ' '.join(re.sub("\s+", " ", text).split())  # Extrem white Space
    text = ' '.join(re.sub("[^A-Za-z^,^.^?^; ]", "", text).split())  # digits
    text = ' '.join(re.sub('-', ' ', text).split())
    text = ' '.join(re.sub('_', ' ', text).split())  # underscore
    return  text

def data_fliter(texts,stopwords):
    # Remove punctuation, special symbols and numbers
    text = [character_filter(item) for item in texts]

    text_filter = []
    for item in text:
        mytext = str(item).lower()  # lower case
        # lemmatization
        tokens = word_tokenize(mytext)  # Participle
        tagged_sent = pos_tag(tokens)  # get word part of speech
        # reduction
        wnl = WordNetLemmatizer()
        lemmas_sent = []
        for tag in tagged_sent:
            wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
        print(lemmas_sent)
        text_filter.append(remove_stopwords(stopwords,lemmas_sent))
        # text_filter.append((tokens))

    return text_filter


# Get the part of speech of a word
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
# Download Data
def loadfile():

    # load stopwords
    stopword_list = stopwords.words("english") #nltk stopword

    # Download Data
    train_data_df = pd.read_excel('./data/train.xlsx')
    train_texts_raw = train_data_df['texts'].tolist()
    train_labels = train_data_df['sentiment'].tolist()

    # data preprocessing
    train_texts = data_fliter(train_texts_raw, stopword_list)
    train_texts_res = []
    train_labels_res = []
    for index,item in enumerate(train_texts):
        if len(item)>0:
            train_texts_res.append(item)
            train_labels_res.append(train_labels[index])
    train_texts_res, train_labels_res = shuffle(train_texts_res, train_labels_res)
    return train_texts_res, train_labels_res#Return data and labels
if __name__ == "__main__":
    loadfile()