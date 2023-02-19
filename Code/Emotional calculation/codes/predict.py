import re
from tqdm import tqdm
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing import sequence
from gensim.models.word2vec import Word2Vec
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings("ignore")
def loadmodel():
    #load training data
    model_word = Word2Vec.load('./model/sentment_Word2Vec.pkl')  # Long loading time
    input_dim = len(model_word.wv.vocab.keys()) + 1  # the number of words +1
    w2dic = {}  # Word Correspondence Index Dictionary
    for i in range(len(model_word.wv.vocab.keys())):
        w2dic[list(model_word.wv.vocab.keys())[i]] = i + 1
    model = load_model('./model/lstm_model.h5')
    return w2dic,model
# remove disabled
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

def predict(w2dic,model,text):
    # load stopwords
    stopword_list = stopwords.words("english")  # Stop words that come with nltk
    train_texts = data_fliter([text], stopword_list)
    new_txt=[]
    data=[]
    for word in train_texts[0]:
        try:
            new_txt.append(w2dic[word])
        except:
            new_txt.append(0)
    data.append(new_txt)
    data=sequence.pad_sequences(data, maxlen=30)
    pre=model.predict_classes(data)[0]
    return pre

if __name__ == '__main__':
    w2dic, model = loadmodel()
    # while 1:
    #     text = input("please enter a sentence： ")
    #     print(predict(w2dic,model,text))
    #
    res = []
    df = pd.read_csv('./data/tweets_en-GB_data.csv')
    texts = df['content'].tolist()
    for text in tqdm(texts):
        res.append(predict(w2dic,model,text))
    df['result'] =res
    df.to_excel('Forecast_results.xlsx',index=0)