import numpy as np
import multiprocessing
from gensim.models.word2vec import Word2Vec

def word2vec_train(X_Vec,voc_dim):
    min_out = 2 # 单词出现频率数
    window_size = 4  #
    cpu_count = multiprocessing.cpu_count()#
    model_word = Word2Vec(size=voc_dim,
                     min_count=min_out,
                     window=window_size,
                     workers=cpu_count,
                     iter=70)
    model_word.build_vocab(X_Vec)#建立初始词典
    model_word.train(X_Vec, total_examples=model_word.corpus_count, epochs=model_word.iter)#开始训练
    model_word.save('./model/sentment_Word2Vec.pkl')#保存模型
    input_dim = len(model_word.wv.vocab.keys()) + 1 #具有词向量的词的数量，不是所有的词都具有词向量，只有符合要求的才可以
    embedding_weights = np.zeros((input_dim, voc_dim))#初始化词向量矩阵
    w2dic={}#初始化词典
    for i in range(len(model_word.wv.vocab.keys())):
        w2dic[list(model_word.wv.vocab.keys())[i]]=i+1#挨个取出模型中每个词并保存
        embedding_weights[i+1, :] = model_word[list(model_word.wv.vocab.keys())[i]]#对应词典中的词取出模型中词的向量并保存
    return input_dim,embedding_weights,w2dic  #input_dim 含有词向量的词的个数embedding_weights每个词的向量，w2dic词典











