# -*- coding: utf-8 -*-
from models import lstm
from word2vec import word2vec_train
from dataset import loadfile
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from plot import *
from keras.utils import np_utils
import time
np.random.seed(1001)

import warnings
warnings.filterwarnings("ignore")
def data2inx(w2indx,X_Vec):
    data = []
    for sentence in X_Vec:
        new_txt = []
        for word in sentence:
            try:
                new_txt.append(w2indx[word])#if in the dictionary, keep it
            except:
                new_txt.append(0)#if not in the dictionary, then 0

        data.append(new_txt)
    return data

def train_model(model, x_train, y_train, x_test, y_test,y_test_):
    time_start = time.time()
    print('Compiling the Model...')
    model.compile(loss='categorical_crossentropy',#hinge
                  optimizer='adam', metrics=['acc'])#set
    print("Train..." )
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch_time,
              verbose=1,validation_data=(x_test, y_test_))#start training
    time_end = time.time()
    print("Evaluate...")
    labels = [0, 1,2,3,4]
    target_names = ["0","1",'2','3','4']
    y_true = [int(item) for item in y_test]
    y_pre = list(model.predict_classes(x_test))
    print(classification_report(y_true, y_pre, labels=labels, target_names=target_names, digits=3))
    print("Training time：",str(round(time_end - time_start,2))+"s")
    #
    #save the model
    model.save('./model/lstm_model.h5')
    #Plot confusion matrix
    plot_confuse(model,x_test,y_test_,target_names,"lstm")


if __name__ =="__main__":
    voc_dim = 100  # Vector dimension
    epoch_time =  10 # epoch
    batch_size = 32  # batch size
    input_length = 30  # Maximum reserved length of text
    print("Loading training data................")
    # train_texts, train_labels,test_texts,test_labels
    x_train,y_train  = loadfile()
    #Used to train word vectors
    input_dim,embedding_weights,w2dic = word2vec_train(x_train,voc_dim,)#Train word vectors

    #Loading completed, text encoding
    train_index = data2inx(w2dic,x_train)#Convert text to an index sequence in a dictionary
    x_train = sequence.pad_sequences(train_index, maxlen=input_length)#Each sentence is processed into the same length, with long truncation and short complement of 0

    # test_index = data2inx(w2dic,x_test)#Convert text to an index sequence in a dictionary
    # x_test = sequence.pad_sequences(test_index, maxlen=input_length)

    x_train_, x_test, y_train_, y_test = train_test_split(x_train, y_train, test_size=0.2)#Split training and testing dataset 8:2
    y_train =  np_utils.to_categorical(y_train, num_classes=5)#Convert the tag to one hot encoding, for example, [0, 0, 1] is neutral, [0, 1, 0] is negative
    y_test_ =  np_utils.to_categorical(y_test, num_classes=5)#Convert the tag to one hot encoding, for example, [0, 0, 1] is neutral, [0, 1, 0] is negative
    #lstm
    model=lstm(input_dim,embedding_weights,voc_dim,input_length)#Model instantiation
    print("Start training lstm model..................")
    train_model(model, x_train, y_train, x_test,y_test, y_test_)
