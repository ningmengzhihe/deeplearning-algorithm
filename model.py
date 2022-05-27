# Deep Learning Tools
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, Conv1D, MaxPool1D
from keras.layers import LSTM, SimpleRNN, GRU, Bidirectional
# Machine Learning Tools
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


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

        elif model_type == 'RNN':
            model = Sequential(name='model')
            model.add(SimpleRNN(input_shape=(max_batch_length, variable_number), units=40, activation='tanh',
                                return_sequences=True))
            model.add(SimpleRNN(units=40, activation='tanh', return_sequences=False))

            model.add(Dense(4))
            model.add(Dense(1))

        elif model_type == 'LSTM':
            model = Sequential(name='model_lstm')
            model.add(LSTM(input_shape=(max_batch_length, variable_number), units=40, activation='tanh',
                                return_sequences=True, name='lstm1'))
            model.add(LSTM(units=40, activation='tanh', return_sequences=False, name='lstm2'))

            model.add(Dense(4, name='dense1'))
            model.add(Dense(1, name='dense2'))

        elif model_type == 'BiLSTM':
            model = Sequential(name='model')
            model.add(Bidirectional(LSTM(units=40, activation='tanh', return_sequences=True),
                                    input_shape=(max_batch_length, variable_number), name='biLSTM1'))
            model.add(Bidirectional(LSTM(units=40, activation='tanh', return_sequences=False), name='biLSTM2'))

            model.add(Dense(4, name='dense1'))
            model.add(Dense(1, name='dense2'))

        elif model_type == 'GRU':
            model = Sequential(name='model_GRU')
            model.add(GRU(input_shape=(max_batch_length, variable_number), units=40, activation='relu',
                          return_sequences=True))
            model.add(GRU(units=40, activation='relu', return_sequences=False))

            model.add(Dense(4))
            model.add(Dense(1))

        return model


class MachineLearingModel:
    def __init__(self):
        pass

    def init_model(self, model_type: str, **kwargs):
        '''
        :param model_type: one of ['LinearRegression', 'Lasso', 'Ridge', 'ElasticNet', 'BayesianRidge', 'KNN', 'SupportVector', 'DecisionTree',
         'RandomForest', 'ExtraTree', 'AdaBoost', 'GBDT', 'Xgboost', 'LightGBM']
        :return:
        '''
        if model_type == 'LinearRegression':
            # model = LinearRegression()
            model = LinearRegression(fit_intercept=kwargs["fit_intercept"])
        elif model_type == 'Lasso':
            # model = Lasso(alpha=0.025)
            model = Lasso(alpha=kwargs["alpha"])
        elif model_type == 'Ridge':
            # model = Ridge(alpha=0.002)
            model = Ridge(alpha=kwargs["alpha"])
        elif model_type == 'ElasticNet':
            # model = ElasticNet(alpha=0.02, l1_ratio=0.7)
            model = ElasticNet(alpha=kwargs["alpha"], l1_ratio=kwargs["l1_ratio"])
        elif model_type == 'BayesianRidge':
            # model = BayesianRidge(n_iter=300, alpha_1=1e-9, alpha_2=1e-9, lambda_1=1e-9, lambda_2=1e-9, alpha_init=0.2, compute_score=False)
            model = BayesianRidge(n_iter=int(kwargs["n_iter"]), alpha_1=kwargs["alpha_1"], alpha_2=kwargs["alpha_2"],
                                  lambda_1=kwargs["lambda_1"], lambda_2=kwargs["lambda_2"], alpha_init=kwargs["alpha_init"],
                                  compute_score=kwargs["compute_score"])
        elif model_type == 'KNN':
            # model = KNeighborsRegressor(n_neighbors=3)
            model = KNeighborsRegressor(n_neighbors=int(kwargs["n_neighbors"]))

        elif model_type == 'SupportVector':
            # model = SVR(kernel='rbf')
            model = SVR(kernel=kwargs["kernel"])

        elif model_type == 'DecisionTree':
            # model = DecisionTreeRegressor(max_depth=5)
            if kwargs["max_depth"] == 0:
                model = DecisionTreeRegressor(max_depth=None)
            else:
                model = DecisionTreeRegressor(max_depth=kwargs["max_depth"])

        elif model_type == 'RandomForest':
            # model = RandomForestRegressor(criterion='squared_error', n_estimators=100, random_state=0)
            if kwargs["max_depth"] == 0:
                model = RandomForestRegressor(n_estimators=int(kwargs["n_estimators"]), max_depth=None)
            else:
                model = RandomForestRegressor(n_estimators=int(kwargs["n_estimators"]), max_depth=kwargs["max_depth"])

        elif model_type == 'ExtraTree':
            # model = ExtraTreesRegressor(n_estimators=50)
            if kwargs["max_depth"] == 0:
                model = ExtraTreesRegressor(n_estimators=int(kwargs["n_estimators"]), max_depth=None)
            else:
                model = ExtraTreesRegressor(n_estimators=int(kwargs["n_estimators"]), max_depth=kwargs["max_depth"])

        elif model_type == 'AdaBoost':
            # model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), n_estimators=100, random_state=None)
            if kwargs["max_depth"] == 0:
                model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=None),
                                          n_estimators=kwargs["n_estimators"],
                                          learning_rate=kwargs["learning_rate"])
            else:
                model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=kwargs["max_depth"]),
                                      n_estimators=kwargs["n_estimators"],
                                      learning_rate=kwargs["learning_rate"])

        elif model_type == 'GBDT':
            # model = GradientBoostingRegressor(n_estimators=50, max_depth=5, learning_rate=0.3)
            model = GradientBoostingRegressor(n_estimators=kwargs["n_estimators"],
                                              max_depth=kwargs["max_depth"],
                                              learning_rate=kwargs["learning_rate"])

        elif model_type == 'Xgboost':
            # model = XGBRegressor(n_estimators=50, max_depth=5, learning_rate=0.3)
            model = XGBRegressor(n_estimators=kwargs["n_estimators"],
                                 max_depth=kwargs["max_depth"],
                                 learning_rate=kwargs["learning_rate"])

        elif model_type == 'LightGBM':
            # model = LGBMRegressor(n_estimators=100, max_depth=5, num_leaves=10, learning_rate=0.1)
            model = LGBMRegressor(n_estimators=kwargs["n_estimators"],
                                  max_depth=kwargs["max_depth"],
                                  num_leaves=kwargs["num_leaves"],
                                  learning_rate=kwargs["learning_rate"])
        return model

    def model_fit_predict(self, model, train_x, train_y, test_x, test_y):
        model.fit(train_x, train_y)
        expected = test_y
        predicted = model.predict(test_x)
        return model, expected, predicted

if __name__ == '__main__':
    # Sample Code for Deep Learning Model
    model_type = 'Conv1d'
    dp_model = DeepLearningModel()
    model = dp_model.init_model(model_type=model_type)

