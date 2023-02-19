import re
import string
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
# data cleaning
def text_cleaning(text):
    """ A function to clean the text data"""

    sample = str(text).lower()
    sample = " ".join([x.lower() for x in sample.split()])
    sample = re.sub(r"\S*https?:\S*", '', sample)  # links and urls
    sample = re.sub(r"(#)?(//)?\s*@\S*?\s*(:| |$)", " ", sample)  # #\@
    sample = re.sub(r"(#)?(//)?\s*#\S*?\s*(:| |$)", " ", sample)  # #\@
    sample = re.sub('\[.*?\]', '', sample)  # text between [square brackets]
    sample = re.sub('\(.*?\)', '', sample)  # text between (parenthesis)
    sample = re.sub('[%s]' % re.escape(string.punctuation), '', sample)  # punctuations
    sample = re.sub('\w*\d\w', '', sample)  # digits with trailing or preceeding text
    sample = re.sub(r'\n', ' ', sample)  # new line character
    sample = re.sub(r'\\n', ' ', sample)  # new line character
    sample = re.sub("[''""...“”‘’…]", '', sample)  # list of quotation marks
    sample = re.sub(r', /<[^>]+>/', '', sample)  # HTML attributes
    sample = re.sub(r"can\'t", "can not", sample)
    sample = re.sub(r"n\'t", " not", sample)
    sample = re.sub(r"\'re", " are", sample)
    sample = re.sub(r"\'s", " is", sample)
    sample = re.sub(r"\'d", " would", sample)
    sample = re.sub(r"\'ll", " will", sample)
    sample = re.sub(r"\'t", " not", sample)
    sample = re.sub(r"\'ve", " have", sample)
    sample = re.sub(r"\'m", " am", sample)
    return sample
if __name__ == '__main__':
    # stop words
    with open('./stop_words.txt', 'r', encoding='utf-8') as f:
        stopwords = [str(line).strip() for line in f.readlines()]

    # load data and filter
    with open('./new_document.txt', 'r', encoding='utf-8') as f:
        data = [text_cleaning(str(line)).strip() for line in f.readlines()]

    # all vocabulary
    allwords = []
    for text in data:
        # Participle
        words = word_tokenize(text)
        print(words)
        for word in words:
            # non-stop-words
            if word not in stopwords:
                allwords.append(word)
    # Statistical word frequency
    word_dict = dict(Counter(allwords).most_common())

    # save result
    pd.DataFrame({'words':list(word_dict.keys()),'fre':list(word_dict.values())}).to_excel('word_frequency1.xlsx',index=0)

    print('Finish')