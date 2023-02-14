import re
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.utils import shuffle
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
        print(lemmas_sent)
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
# 加载数据
def loadfile():

    # 加载停用词
    stopword_list = stopwords.words("english") #nltk自带的停用词

    # 加载数据
    train_data_df = pd.read_excel('./data/train.xlsx')
    train_texts_raw = train_data_df['texts'].tolist()
    train_labels = train_data_df['sentiment'].tolist()

    # 数据预处理
    train_texts = data_fliter(train_texts_raw, stopword_list)
    train_texts_res = []
    train_labels_res = []
    for index,item in enumerate(train_texts):
        if len(item)>0:
            train_texts_res.append(item)
            train_labels_res.append(train_labels[index])
    train_texts_res, train_labels_res = shuffle(train_texts_res, train_labels_res)
    return train_texts_res, train_labels_res#返回数据与标签
if __name__ == "__main__":
    loadfile()