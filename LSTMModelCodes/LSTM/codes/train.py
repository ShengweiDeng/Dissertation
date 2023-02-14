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
                new_txt.append(w2indx[word])#词典有的词保留
            except:
                new_txt.append(0)#词典没有的词换为0

        data.append(new_txt)
    return data

def train_model(model, x_train, y_train, x_test, y_test,y_test_):
    time_start = time.time()
    print('Compiling the Model...')
    model.compile(loss='categorical_crossentropy',#hinge
                  optimizer='adam', metrics=['acc'])#设置训练方式，包括损失计算方法，优化器以及模型评估方法（准确率）
    print("Train..." )
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch_time,
              verbose=1,validation_data=(x_test, y_test_))#开始训练
    time_end = time.time()
    print("Evaluate...")
    labels = [0, 1,2,3,4]
    target_names = ["0","1",'2','3','4']
    y_true = [int(item) for item in y_test]
    y_pre = list(model.predict_classes(x_test))
    print(classification_report(y_true, y_pre, labels=labels, target_names=target_names, digits=3))
    print("训练用时：",str(round(time_end - time_start,2))+"s")
    #
    #保存模型
    model.save('./model/lstm_model.h5')
    #绘制混淆矩阵
    plot_confuse(model,x_test,y_test_,target_names,"lstm")


if __name__ =="__main__":
    voc_dim = 100  # 词向量向量维度
    epoch_time =  10 # epoch
    batch_size = 32  # batch size
    input_length = 30  # 文本最大保留长度
    print("加载训练数据................")
    # train_texts, train_labels,test_texts,test_labels
    x_train,y_train  = loadfile()
    #用于训练词向量，
    input_dim,embedding_weights,w2dic = word2vec_train(x_train,voc_dim,)#训练词向量

    #加载完成，文本编码
    train_index = data2inx(w2dic,x_train)#将文本转化为字典中的索引序列
    x_train = sequence.pad_sequences(train_index, maxlen=input_length)#将每句话处理成同样的长度，长的截断，短的补充0

    # test_index = data2inx(w2dic,x_test)#将文本转化为字典中的索引序列
    # x_test = sequence.pad_sequences(test_index, maxlen=input_length)#将每句话处理成同样的长度，长的截断，短的补充0

    x_train_, x_test, y_train_, y_test = train_test_split(x_train, y_train, test_size=0.2)#分割训练与验证数据集8:2
    y_train =  np_utils.to_categorical(y_train, num_classes=5)#将标签转化为one hot 编码的方式，例如[0，0，1]为中性，[0，1，0]为消极
    y_test_ =  np_utils.to_categorical(y_test, num_classes=5)#与上同理
    #lstm
    model=lstm(input_dim,embedding_weights,voc_dim,input_length)#模型实例化
    print("开始训练lstm模型..................")
    train_model(model, x_train, y_train, x_test,y_test, y_test_)
