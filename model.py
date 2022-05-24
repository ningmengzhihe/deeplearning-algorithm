# Standard library import

# Deep Learning Tools
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, Conv1D, MaxPool1D
from keras.layers import LSTM, SimpleRNN, GRU, Bidirectional


class DeepLearningModel:
    def __init__(self):
        pass

    def init_model(self, model_type: str):
        max_batch_length = 263
        variable_number = 19
        if model_type == 'Conv1d':
            model = Sequential(name='model_conv1d')
            model.add(Conv1D(filters=256, kernel_size=2, activation='relu', input_shape=(max_batch_length, variable_number),
                                    name='conv1d1'))
            model.add(MaxPool1D(8, name='maxpool1'))
            model.add(Conv1D(filters=128, kernel_size=2, activation='relu', name='conv1d2'))
            model.add(MaxPool1D(8, name='maxpool2'))
            model.add(Conv1D(filters=64, kernel_size=2, activation='relu', name='conv1d3'))
            model.add(MaxPool1D(2, name='maxpool3'))

            model.add(Flatten(name='flatten'))

            model.add(Dense(16, activation='linear', name='dense1'))
            model.add(Dense(1, activation='linear', name='dense2'))

        elif model_type == 'Conv2d':
            model = Sequential(name='model_conv2d')

            model.add(Conv2D(filters=16, kernel_size=[2, 2], padding='same',
                             input_shape=(max_batch_length, variable_number, 1)))
            model.add(Activation('relu'))
            model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

            model.add(Conv2D(filters=32, kernel_size=[2, 2], padding='same'))
            model.add(Activation('relu'))
            model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

            model.add(Conv2D(filters=64, kernel_size=[2, 2], padding='same'))
            model.add(Activation('relu'))
            model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

            model.add(Conv2D(filters=16, kernel_size=[2, 2], padding='same'))
            model.add(Activation('relu'))
            model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

            model.add(Flatten())
            model.add(Dense(64))
            model.add(Activation('relu'))

            model.add(Dense(16))
            model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('linear'))

        elif model_type == 'LSTM':
            model = Sequential(name='model_lstm')
            model.add(LSTM(input_shape=(max_batch_length, variable_number), units=40, activation='tanh',
                                return_sequences=True, name='lstm1'))
            model.add(LSTM(units=40, activation='tanh', return_sequences=False, name='lstm2'))

            model.add(Dense(4, name='dense1'))
            model.add(Dense(1, name='dense2'))

        elif model_type == 'RNN':
            model = Sequential(name='model')
            model.add(SimpleRNN(input_shape=(max_batch_length, variable_number), units=40, activation='tanh',
                                return_sequences=True))
            model.add(SimpleRNN(units=40, activation='tanh', return_sequences=False))

            model.add(Dense(4))
            model.add(Dense(1))

        elif model_type == 'GRU':
            model = Sequential(name='model_GRU')
            model.add(GRU(input_shape=(max_batch_length, variable_number), units=40, activation='relu',
                          return_sequences=True))
            model.add(GRU(units=40, activation='relu', return_sequences=False))

            model.add(Dense(4))
            model.add(Dense(1))
        elif model_type == 'BiLSTM':
            model = Sequential(name='model')
            model.add(Bidirectional(LSTM(units=40, activation='tanh', return_sequences=True),
                                    input_shape=(max_batch_length, variable_number), name='biLSTM1'))
            model.add(Bidirectional(LSTM(units=40, activation='tanh', return_sequences=False), name='biLSTM2'))

            model.add(Dense(4, name='dense1'))
            model.add(Dense(1, name='dense2'))

        return model


if __name__ == '__main__':
    model_type = 'Conv1d'
    dp_model = DeepLearningModel()
    model = dp_model.init_model(model_type=model_type)

