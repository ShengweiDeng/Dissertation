import numpy as np
import multiprocessing
from gensim.models.word2vec import Word2Vec

def word2vec_train(X_Vec,voc_dim):
    min_out = 2 # word frequency
    window_size = 4  #
    cpu_count = multiprocessing.cpu_count()#
    model_word = Word2Vec(size=voc_dim,
                     min_count=min_out,
                     window=window_size,
                     workers=cpu_count,
                     iter=70)
    model_word.build_vocab(X_Vec)#build initial dictionary
    model_word.train(X_Vec, total_examples=model_word.corpus_count, epochs=model_word.iter)#start training
    model_word.save('./model/sentment_Word2Vec.pkl')#save model
    input_dim = len(model_word.wv.vocab.keys()) + 1 #The number of words with word vectors, not all words have word vectors, only those that meet the requirements
    embedding_weights = np.zeros((input_dim, voc_dim))#Initialize word vector matrix
    w2dic={}#Initialize the dictionary
    for i in range(len(model_word.wv.vocab.keys())):
        w2dic[list(model_word.wv.vocab.keys())[i]]=i+1#Take out each word in the model one by one and save it
        embedding_weights[i+1, :] = model_word[list(model_word.wv.vocab.keys())[i]]#Corresponding to the word in the dictionary, take out the vector of the word in the model and save it
    return input_dim,embedding_weights,w2dic  #input_dim The number of words containing word vectors embedding_weights The vector of each word, w2dic dictionary











