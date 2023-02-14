from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
import warnings
warnings.filterwarnings("ignore")


def lstm(input_dim,embedding_weights,voc_dim,input_length):
    model = Sequential()
    model.add(Embedding(output_dim=voc_dim,#指的是输出数据的维度，意思是经过Embedding层降维后的数据由多少个元素组成
                        input_dim=input_dim,#指的是输入数据的维度，意思就是这一行数据是由多少个元素组成的
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))#指的是输入数据的长度，意思是输入数据的长度是多少
    model.add(LSTM(128, activation='tanh',return_sequences=True))#第一层lstm
    model.add(Dropout(0.3))  # dropout 随机丢弃模型中的参数，防止过拟合
    model.add(LSTM(32, activation='tanh'))#第二层lstm
    model.add(Dropout(0.3))#dropout 随机丢弃模型中的参数，防止过拟合
    model.add(Dense(5))#全连接层
    model.add(Activation('softmax'))#最后一层
    print("打印模型")
    model.summary()
    return model


