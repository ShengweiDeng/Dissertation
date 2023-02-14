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
    #加载训练数据
    model_word = Word2Vec.load('./model/sentment_Word2Vec.pkl')  # 加载时间比较长
    input_dim = len(model_word.wv.vocab.keys()) + 1  # 词的个数+1
    w2dic = {}  # 词对应索引 字典
    for i in range(len(model_word.wv.vocab.keys())):
        w2dic[list(model_word.wv.vocab.keys())[i]] = i + 1
    model = load_model('./model/lstm_model.h5')
    return w2dic,model
# 去除停用粗
def remove_stopwords(stopwords,word_list):
    res = []
    for item in word_list:
        if item not in stopwords:
            res.append(item)
    return res
# 文本过滤
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
    # 去除标点符号、特殊符号和数字
    text = [character_filter(item) for item in texts]

    text_filter = []
    for item in text:
        mytext = str(item).lower()  # 小写
        # 词形还原
        tokens = word_tokenize(mytext)  # 分词
        tagged_sent = pos_tag(tokens)  # 获取单词词性
        # 还原
        wnl = WordNetLemmatizer()
        lemmas_sent = []
        for tag in tagged_sent:
            wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
        text_filter.append(remove_stopwords(stopwords,lemmas_sent))
        # text_filter.append((tokens))

    return text_filter


# 获取单词的词性
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
    # 加载停用词
    stopword_list = stopwords.words("english")  # nltk自带的停用词
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
    #     text = input("请输入句子： ")
    #     print(predict(w2dic,model,text))
    #
    res = []
    df = pd.read_csv('./data/tweets_en-GB_data.csv')
    texts = df['content'].tolist()
    for text in tqdm(texts):
        res.append(predict(w2dic,model,text))
    df['result'] =res
    df.to_excel('预测结果.xlsx',index=0)