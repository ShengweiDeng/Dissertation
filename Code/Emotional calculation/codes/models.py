from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
import warnings
warnings.filterwarnings("ignore")


def lstm(input_dim,embedding_weights,voc_dim,input_length):
    model = Sequential()
    model.add(Embedding(output_dim=voc_dim,#Refers to the dimension of the output data, which means how many elements the data after the dimensionality reduction of the Embedding layer consists of
                        input_dim=input_dim,#Refers to the dimension of the input data, which means how many elements this row of data is composed of
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))#Refers to the length of the input data, which means the length of the input data
    model.add(LSTM(128, activation='tanh',return_sequences=True))#The first layer lstm
    model.add(Dropout(0.3))  # dropout Randomly discard parameters in the model to prevent overfitting
    model.add(LSTM(32, activation='tanh'))#The second layer lstm
    model.add(Dropout(0.3))#dropout Randomly discard parameters in the model to prevent overfitting
    model.add(Dense(5))#fully connected layer
    model.add(Activation('softmax'))#last layer
    print("print model")
    model.summary()
    return model


