# System Tools
import time
# Data Analysis Tools
import numpy as np
import pandas as pd
# Machine Learning Tools
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Deep Learning Tools
from tensorflow.keras.optimizers import SGD, Adam, Adagrad, Adamax, Adadelta, RMSprop
# Streamlit Tools
import streamlit as st
# UserDefine Class
from model import DeepLearningModel, MachineLearingModel

# 初始化
config_dp = {
    # "dataset": "PHM 2016 Challenge",
    "optimizer": {"optimizer": "Adam",
                  "kwargs": {"beta1": 0.9, "beta2": 0.999, "epsilon": 1e-7}},
    "batch_size": 200,
    "epochs": 2,  # 400,
    "learning_rate": 0.001,
    "loss_function": "mean_squared_error",
    "batch_size_max_value": 798,
    "model": 'Conv1d',
    "graph_granularity": 1,
    "input_shape": (263, 19)
}
config_ml = {
    # "dataset": "PHM 2016 Challenge",
    "model": 'LinearRegression',
    "kwargs": {
        "fit_intercept": True
    }
}
option_layer_initializer = ['Constant', 'GlorotNormal', 'GlorotUniform', 'HeNormal', 'Identity', 'LeCunNormal', 'Ones',
                             'Orthogonal', 'RandomNormal', 'RandomUniform', 'TruncatedNormal', 'VarianceScaling', 'Zeros']
option_activation = ['none', 'Relu', 'Elu', 'Sigmoid', 'Softmax', 'Softplus', 'Tanh', 'Selu', 'Relu6']
option_model_dp = ['Conv1d', 'Conv2d', 'RNN', 'LSTM', 'BiLSTM', 'GRU']
option_model_ml = ['LinearRegression', 'Lasso', 'Ridge', 'ElasticNet', 'BayesianRidge', 'KNN', 'SupportVector', 'DecisionTree',
                   'RandomForest', 'ExtraTree', 'AdaBoost', 'GBDT', 'Xgboost', 'LightGBM']


# load data function
@st.cache
def load_dataset(dataset_name: str, model_type='Conv1d'):
    '''
    dataset_name:str: st.selectbox
    model_type: Default is a deeplearning model
    :return:
    '''

    if dataset_name == 'PHM 2016 Challenge':
        if model_type in option_model_dp:
            # X_train = np.load("/Users/lihan/Workspace/data phm 2016/X_train_r_modeI_chamber4_mm.npy")
            # y_train = np.load("/Users/lihan/Workspace/data phm 2016/y_train_modeI_chamber4_mm.npy")
            # X_test = np.load("/Users/lihan/Workspace/data phm 2016/X_test_r_modeI_chamber4_mm.npy")
            # y_test = np.load("/Users/lihan/Workspace/data phm 2016/y_test_modeI_chamber4_mm.npy")
            X_train = np.load("./data/X_train_r_modeI_chamber4_mm.npy")
            y_train = np.load("./data/y_train_modeI_chamber4_mm.npy")
            X_test = np.load("./data/X_test_r_modeI_chamber4_mm.npy")
            y_test = np.load("./data/y_test_modeI_chamber4_mm.npy")
            return X_train, y_train, X_test, y_test

        elif model_type in option_model_ml:
            # 读取时的用法
            train_x = pd.read_csv("./data/train_x_mean_modeI_chamber4_mm.csv").set_index('WAFER_ID')
            train_y = pd.read_csv("./data/train_y_mean_modeI_chamber4_mm.csv").set_index('WAFER_ID')
            test_x = pd.read_csv("./data/test_x_mean_modeI_chamber4_mm.csv").set_index('WAFER_ID')
            test_y = pd.read_csv("./data/test_y_mean_modeI_chamber4_mm.csv").set_index('WAFER_ID')
            return train_x, train_y, test_x, test_y


@st.cache
def batch_data_generate(X, y, batch_size):
    '''

    :param X: np.array
    :param y: np.array
    :return: batch_data_list,e.g [(X_1, y_1), ..., (X_n, y_n)]
    '''
    N = y.shape[0] # 数据总数
    step = int(np.ceil(N/batch_size))
    batch_data_list = []
    for i in range(0, step-1):
        batch_data_list.append((X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size]))
    batch_data_list.append((X[(step-1)*batch_size:], y[(step-1)*batch_size:]))
    return batch_data_list


def st_conv1d(filters, key_name):
    st.number_input(label='Filters', value=filters, format='%d', key=f'num_filters_{key_name}',
                    help='卷积核的数目（即输出的维度）')
    st.number_input(label='Kernel Size', value=2, format='%d', key=f'num_ker_size_{key_name}',
                    help='整数或由单个整数构成的list/tuple，卷积核的空域或时域窗长度')
    st.number_input(label='Strides', value=1, format='%d', key=f'num_stride_{key_name}',
                    help='整数或由单个整数构成的list/tuple，为卷积的步长。')
    st.selectbox(label='Padding',
                 options=['Valid', 'Same', 'Casual'],
                 index=0, key=f'sel_pad_{key_name}',
                 help='补0策略，“valid”意味着没有填充。 “same”导致在输入的左/右或上/下均匀填充零，以使输出具有与输入相同的高度/宽度尺寸。 “casual”导致因果（扩张）卷积')
    st.info('ℹ️ Careful: There is no default activation on Conv1D layers')
    st.selectbox(label='Activation',
                 options=option_activation,
                 index=1, key=f'sel_active_{key_name}',
                 help='激活函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）')
    st.checkbox(label='Use a Bias', value=True, key=f'check_bias_{key_name}',
                help='布尔值，是否使用偏置项')
    st.selectbox(label='Bias Initializer',
                 options=option_layer_initializer,
                 index=12, key=f'sel_bias_{key_name}',
                 help='偏置向量的初始化器')
    st.selectbox(label='Kernel Initializer',
                 options=option_layer_initializer,
                 index=2, key=f'sel_ker_init_{key_name}',
                 help='内核权重矩阵的初始化方法')


def st_conv2d(filters, key_name):
    st.number_input(label='Filters', value=filters, format='%d', key=f'num_filters_{key_name}',
                    help='卷积核的数目（即输出的维度）')
    st.number_input(label='Kernel Size', value=2, format='%d', key=f'num_ker_size_{key_name}',
                    help='单个整数或由两个整数构成的list/tuple，卷积核的宽度和长度。如为单个整数，则表示在各个空间维度的相同长度')
    st.number_input(label='Strides', value=1, format='%d', key=f'num_stride_{key_name}',
                    help='单个整数或由两个整数构成的list/tuple，为卷积的步长。如为单个整数，则表示在各个空间维度的相同步长')
    st.selectbox(label='Padding',
                 options=['Valid', 'Same'],
                 index=1, key=f'sel_pad_{key_name}',
                 help='补0策略，为“valid”, “same” 。“valid”代表只进行有效的卷积，即对边界数据不处理。“same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同')
    st.info('ℹ️ Careful: There is no default activation on Conv2D layers')
    st.selectbox(label='Activation',
                 options=option_activation,
                 index=1, key=f'sel_active_{key_name}',
                 help='激活函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）')
    st.checkbox(label='Use a Bias', value=True, key=f'check_bias_{key_name}',
                help='布尔值，是否使用偏置项')
    st.selectbox(label='Bias Initializer',
                 options=option_layer_initializer,
                 index=12, key=f'sel_bias_{key_name}',
                 help='偏置向量的初始化器')
    st.selectbox(label='Kernel Initializer',
                 options=option_layer_initializer,
                 index=2, key=f'sel_ker_init_{key_name}',
                 help='内核权重矩阵的初始化方法')


def st_maxpool1d(poolsize, key_name):
    st.number_input(label='Pool Size', value=poolsize, format='%d', key=f'num_pool_{key_name}',
                    help='整数，池化窗口大小')
    st.number_input(label='Strides', value=poolsize, format='%d', key=f'num_stride_{key_name}',
                    help='整数或None，下采样因子，例如设2将会使得输出shape为输入的一半，若为None则默认值为pool_size')
    st.selectbox(label='Padding',
                 options=['Valid', 'Same'],
                 index=0, key=f'sel_pad_{key_name}',
                 help=' “Valid”意味着没有填充。 “Same”导致在输入的左/右或上/下均匀填充，以使输出具有与输入相同的高度/宽度尺寸。')


def st_maxpool2d(key_name):
    st.number_input(label='Pool Size', value=2, format='%d', key=f'num_pool_{key_name}',
                    help='整数或长为2的整数tuple，代表在两个方向（竖直，水平）上的下采样因子，如取（2，2）将使图片在两个维度上均变为原长的一半。为整数意为各个维度值相同且为该数字')
    st.number_input(label='Strides', value=2, format='%d', key=f'num_stride_{key_name}',
                    help='整数或长为2的整数tuple，或者None，步长值')
    st.selectbox(label='Padding',
                 options=['Valid', 'Same'],
                 index=1, key=f'sel_pad_{key_name}',
                 help='“Valid”意味着没有填充。 “Same”导致在输入的左/右或上/下均匀填充，以使输出具有与输入相同的高度/宽度尺寸。')


def st_rnn(units, activation, return_seq, key_name):
    st.number_input(label='Units', value=units, min_value=1, format='%d', key=f'num_units_{key_name}',
                    help='正整数，输出空间的维度')
    st.selectbox(label='Activation',
                 options=option_activation,
                 index=option_activation.index(activation),
                 key=f'sel_active_{key_name}',
                 help='要使用的激活函数。 如果None，则不应用激活（即“线性”激活：a(x) = x）')
    st.checkbox(label='Use a Bias', value=True, key=f'check_bias_{key_name}',
                help='布尔值，layer是否使用偏置向量')
    st.selectbox(label='Bias Initializer',
                 options=option_layer_initializer,
                 index=12, key=f'sel_bias_{key_name}',
                 help='偏置向量的初始化器。')
    st.selectbox(label='Kernel Initializer',
                 options=option_layer_initializer,
                 index=2, key=f'sel_ker_init_{key_name}',
                 help='内核权重矩阵的初始化器，用于输入的线性变换。')
    st.checkbox(label='Return Sequences', value=return_seq, key=f'check_return_seq_{key_name}',
                help='布尔值。 是返回输出序列中的最后一个输出，还是返回完整序列')
    st.selectbox(label='Recurrent Initializer',
                 options=option_layer_initializer,
                 index=7, key=f'sel_bias_{key_name}',
                 help='循环内核权重矩阵的初始化器，用于循环状态的线性变换')


def st_lstm(units, activation, recurrent_activation, return_seq, key_name):
    st.number_input(label='Units', value=units, min_value=1, format='%d', key=f'num_units_{key_name}',
                    help='正整数，输出空间的维度')
    st.selectbox(label='Activation',
                 options=option_activation,
                 index=option_activation.index(activation),
                 key=f'sel_active_{key_name}',
                 help='激活函数，如果None，则不应用激活（即“线性”激活：a(x) = x）')
    st.selectbox(label='Recurrent Activation',
                 options=option_activation,
                 index=option_activation.index(recurrent_activation),
                 key=f'sel_recu_active_{key_name}',
                 help='用于循环步骤的激活函数。如果None，则不应用激活（即“线性”激活：a(x) = x）')
    st.checkbox(label='Use a Bias', value=True, key=f'check_bias_{key_name}',
                help='布尔值，layer是否使用偏置向量')
    st.selectbox(label='Bias Initializer',
                 options=option_layer_initializer,
                 index=12, key=f'sel_bias_{key_name}',
                 help='偏置向量的初始化器')
    st.selectbox(label='Kernel Initializer',
                 options=option_layer_initializer,
                 index=2, key=f'sel_ker_init_{key_name}',
                 help='内核权重矩阵的初始化器，用于输入的线性变换')
    st.checkbox(label='Return Sequences', value=return_seq, key=f'check_return_seq_{key_name}',
                help='布尔值，是否返回最后的输出，在输出序列或完整序列中。')
    st.selectbox(label='Recurrent Initializer',
                 options=option_layer_initializer,
                 index=7, key=f'sel_bias_{key_name}',
                 help='循环核的初始化方法')


def st_bilstm(units, activation, recurrent_activation, return_seq, key_name):
    st.number_input(label='Units', value=units, min_value=1, format='%d', key=f'num_units_{key_name}',
                    help='正整数，输出空间的维度')
    st.selectbox(label='Activation',
                 options=option_activation,
                 index=option_activation.index(activation),
                 key=f'sel_active_{key_name}',
                 help='激活函数，如果None，则不应用激活（即“线性”激活：a(x) = x）')
    st.selectbox(label='Recurrent Activation',
                 options=option_activation,
                 index=option_activation.index(recurrent_activation),
                 key=f'sel_recu_active_{key_name}',
                 help='用于循环步骤的激活函数。如果None，则不应用激活（即“线性”激活：a(x) = x）')
    st.checkbox(label='Use a Bias', value=True, key=f'check_bias_{key_name}',
                help='布尔值，layer是否使用偏置向量')
    st.selectbox(label='Bias Initializer',
                 options=option_layer_initializer,
                 index=12, key=f'sel_bias_{key_name}',
                 help='偏置向量的初始化器')
    st.selectbox(label='Kernel Initializer',
                 options=option_layer_initializer,
                 index=2, key=f'sel_ker_init_{key_name}',
                 help='内核权重矩阵的初始化器，用于输入的线性变换')
    st.checkbox(label='Return Sequences', value=return_seq, key=f'check_return_seq_{key_name}',
                help='布尔值，是否返回最后的输出，在输出序列或完整序列中。')
    st.selectbox(label='Recurrent Initializer',
                 options=option_layer_initializer,
                 index=7, key=f'sel_bias_{key_name}',
                 help='循环核的初始化方法')
    st.selectbox(label='Merge_mode',
                 options=['Sum', 'Mul', 'Concat', 'Ave', 'none'],
                 index=2, key=f'sel_bias_{key_name}',
                 help='组合前向和后向RNN的输出的模式，如果为None，则不会合并输出，它们将作为列表返回')


def st_gru(units, activation, recurrent_activation, return_seq, key_name):
    st.number_input(label='Units', value=units, min_value=1, format='%d', key=f'num_units_{key_name}',
                    help='正整数，输出空间的维度')
    st.selectbox(label='Activation',
                 options=option_activation,
                 index=option_activation.index(activation),
                 key=f'sel_active_{key_name}',
                 help='激活函数，如果None，则不应用激活（即“线性”激活：a(x) = x）')
    st.selectbox(label='Recurrent Activation',
                 options=option_activation,
                 index=option_activation.index(recurrent_activation),
                 key=f'sel_recu_active_{key_name}',
                 help='用于循环步骤的激活函数，如果None，则不应用激活（即“线性”激活：a(x) = x）')
    st.checkbox(label='Use a Bias', value=True, key=f'check_bias_{key_name}',
                help='布尔值，图层是否使用偏置向量')
    st.selectbox(label='Bias Initializer',
                 options=option_layer_initializer,
                 index=12, key=f'sel_bias_{key_name}',
                 help='偏置向量的初始化器')
    st.selectbox(label='Kernel Initializer',
                 options=option_layer_initializer,
                 index=2, key=f'sel_ker_init_{key_name}',
                 help='应用于核权重矩阵的正则化函数')
    st.checkbox(label='Return Sequences', value=return_seq, key=f'check_return_seq_{key_name}',
                help='布尔值，是返回输出序列中的最后一个输出，还是返回完整序列。')
    st.selectbox(label='Recurrent Initializer',
                 options=option_layer_initializer,
                 index=7, key=f'sel_bias_{key_name}',
                 help='循环核的初始化方法')
    st.checkbox(label='Reset After',
                value=True,
                key=f'check_reset_{key_name}',
                help='布尔值，是否返回除了输出之外的最后一个状态')


def st_flatten(info):
    # st.info('ℹ️ Will flatten input. i.g.: [28, 28] => [784]')
    st.info(info)


def st_dense(units, activation, key_name):
    st.number_input(label='Units', value=units, min_value=1, key=f'num_units_{key_name}',
                    help='正整数，输出空间的维度')
    st.selectbox(label='Activation',
                 options=option_activation,
                 index=option_activation.index(activation),
                 key=f'sel_active_{key_name}',
                 help='要使用的激活功能，如果None，则不应用任何激活（即“线性”激活：a(x) = x）')
    st.selectbox(label='Kernel Initializer',
                 options=option_layer_initializer,
                 index=2, key=f'sel_ker_init_{key_name}',
                 help='内核权重矩阵的初始化器')
    st.checkbox(label='Use a Bias', value=True, key=f'check_bias_{key_name}',
                help='布尔值，图层是否使用偏置向量')
    st.selectbox(label='Bias Initializer',
                 options=option_layer_initializer,
                 index=12, key=f'sel_bias_{key_name}',
                 help='偏置向量的初始化器')


st.set_page_config(page_title="Deep Learning Model & Machine Learning Model", page_icon="🎈", layout="wide")


st.session_state = config_dp

with st.sidebar:
    # 菜单：读取数据/选择model/调节超参数/设置优化器
    with st.expander('Dataset'):
        # 选择数据集
        dataset_option = st.selectbox(
            label='How would you like to be contacted?',
            options=['PHM 2016 Challenge', 'MNIST', 'FASHION_MNIST', 'CIFAR10', 'QUICK_DRAW10', 'QUICK_DRAW_30'])
        st.session_state['dataset'] = dataset_option
        if dataset_option == 'PHM 2016 Challenge':
            X_train, y_train, X_test, y_test = load_dataset(dataset_name=dataset_option)
            st.write('Total')
            st.write(y_train.shape[0] + y_test.shape[0])
            st.write('Input shape')
            st.write(X_train.shape[1:])
            st.write('Train set')
            st.write(y_train.shape[0])
            st.write('Test set')
            st.write(y_test.shape[0])
            # 更新session
            st.session_state['batch_size_max_value'] = y_train.shape[0]
        else:
            st.write('TBC')

    with st.expander('Choose a model'):
        model_radio = st.radio(label='Which kind of model would you like?',
                               options=['Deep Learning Model', 'Traditional Machine Learning Model'])
        if model_radio == 'Deep Learning Model':
            # 初始化st.session_state
            st.session_state = config_dp
            # 选择一种深度学习模型
            model_select = st.selectbox(
                label='Which model would you like to use?',
                options=option_model_dp,
                index=0)
        else:
            # 初始化st.session_state
            st.session_state = config_ml
            # 选择一种传统的机器学习模型
            model_select = st.selectbox(
                label='Which model would you like to use?',
                options=option_model_ml,
                index=0)
        # 更新session
        st.session_state['model'] = model_select

    if st.session_state['model'] in option_model_dp:
        # 深度学习模型
        with st.expander('Hyperparameters'):
            # 调节超参数
            batch_size = st.number_input(label='BatchSize:',
                                         min_value=1,
                                         max_value=st.session_state['batch_size_max_value'],
                                         step=1,
                                         value=st.session_state['batch_size'],
                                         help='整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步')
            epochs = st.number_input(label='Epochs',
                                     min_value=1,
                                     max_value=1000,
                                     step=10,
                                     value=st.session_state['epochs'],
                                     help='整数，训练的轮数，训练数据将会被遍历nb_epoch次')
            learning_rate = st.number_input(label='LearningRate',
                                            min_value=0.001,
                                            max_value=0.999,
                                            step=0.001,
                                            value=st.session_state['learning_rate'],
                                            format='%.3f',
                                            help='学习率')
            # 更新session
            st.session_state['batch_size'] = batch_size
            st.session_state['epochs'] = epochs
            st.session_state['learning_rate'] = learning_rate

        with st.expander('Optimizer'):
            # 设置损失函数
            loss_option = st.selectbox(label='Loss Function',
                                       options=['MeanSquaredError'],
                                       index=0,
                                       help='损失函数')
            # 更新loss_function
            loss_dict = {'MeanSquaredError': 'mean_squared_error'}
            st.session_state['loss_function'] = loss_dict[loss_option]

            # 设置优化器
            optimizer_option = st.selectbox(label='Optimizer Function',
                                            options=['Sgd', 'Momentum', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Rmsprop'],
                                            index=4,
                                            help='优化器')
            if optimizer_option == 'Momentum':
                momentum = st.number_input(label='Momentum', min_value=0.0, value=0.0, step=0.01,
                                           help='动量参数，大于0的浮点数，加速相关方向的梯度下降并抑制振荡')
                use_nesterov = st.checkbox(label='useNesterov', value=False,
                                           help='确定是否使用Nesterov动量，布尔值')
            elif optimizer_option == 'Adagrad':
                initial_accumulator_value = st.number_input(label='InitialAccumulatorValue',
                                                            value=0.1, min_value=0.0, format="%.3f",
                                                                help='浮点值，累加器的起始值（每个参数的动量值），必须是非负数')
            elif optimizer_option == 'Adadelta':
                rho = st.number_input(label='Rho', value=0.95, min_value=0.0, step=0.01,
                                      help='衰减率')
                epsilon = st.number_input(label='Epsilon', value=1e-7, min_value=0.0, step=1e-7, format="%g",
                                          help='用于保持数值稳定性的小浮点值')
            elif optimizer_option == 'Adam':
                beta1 = st.number_input(label='Beta1', value=0.9, min_value=0.001, max_value=0.999, step=0.05)
                beta2 = st.number_input(label='Beta2', value=0.999, min_value=0.001, max_value=0.999, step=0.005,
                                        format="%.3f")
                epsilon = st.number_input(label='Epsilon', value=1e-7, min_value=0.0, step=1e-7, format="%g")
            elif optimizer_option == 'Adamax':
                beta1 = st.number_input(label='Beta1', value=0.9, min_value=0.001, max_value=0.999, step=0.05,
                                        help='一阶矩估计的指数衰减率')
                beta2 = st.number_input(label='Beta2', value=0.999, min_value=0.001, max_value=0.999, step=0.005,
                                        format="%.3f", help='二阶矩估计的指数衰减率')
                epsilon = st.number_input(label='Epsilon', value=1e-7, min_value=0.0, step=1e-7, format="%g",
                                          help='数值稳定性的小常数')
                decay = st.number_input(label='Decay', value=0.0, min_value=0.0, step=0.01,
                                        help='大于0的浮点数，每次更新后的学习率衰减值')
            elif optimizer_option == 'Rmsprop':
                decay = st.number_input(label='Decay', value=0.0, min_value=0.0, step=0.01,
                                        help='大于0的浮点数，每次更新后的学习率衰减值')
                momentum = st.number_input(label='Momentum', value=0.0, min_value=0.0, step=0.01,
                                           help='动量参数，大于0的浮点数，加速相关方向的梯度下降并抑制振荡')
                epsilon = st.number_input(label='Epsilon', value=1e-7, min_value=0.0, step=1e-7, format="%g",
                                          help='数值稳定性的小常数')
                centered = st.checkbox(label='centered', value=False,
                                       help='布尔值。 如果为 True，则通过梯度的估计方差对梯度进行归一化； 如果为 False，则通过非居中的第二时刻。 将此设置为 True 可能有助于训练，但在计算和内存方面稍贵一些。 默认为假')
            # 显示学习速率
            st.number_input(label='LearningRate',
                            # value=lr_show,
                            value=st.session_state['learning_rate'],
                            disabled=True)
            st.info('ℹ️ LR is set in the "Hyperparameters tab"')

            if optimizer_option == 'Sgd':
                st.session_state['optimizer'] = {'optimizer': 'Sgd',
                                                 'kwargs': {}}
            elif optimizer_option == 'Momentum':
                st.session_state['optimizer'] = {'optimizer': 'Momentum',
                                                 'kwargs': {'momentum': momentum, 'use_nesterov': use_nesterov}}
            elif optimizer_option == 'Adagrad':
                st.session_state['optimizer'] = {'optimizer': 'Adagrad',
                                                 'kwargs': {'initial_accumulator_value': initial_accumulator_value}}
            elif optimizer_option == 'Adadelta':
                st.session_state['optimizer'] = {'optimizer': 'Adadelta',
                                                 'kwargs': {'rho': rho, 'epsilon': epsilon}}
            elif optimizer_option == 'Adam':
                st.session_state['optimizer'] = {'optimizer': 'Adam',
                                                 'kwargs': {'beta1': beta1, 'beta2': beta2, 'epsilon': epsilon}}
            elif optimizer_option == 'Adamax':
                st.session_state['optimizer'] = {'optimizer': 'Adamax',
                                                 'kwargs': {'beta1': beta1, 'beta2': beta2, 'epsilon': epsilon, 'decay': decay}}
            elif optimizer_option == 'Rmsprop':
                st.session_state['optimizer'] = {'optimizer': 'Rmsprop',
                                                 'kwargs': {'decay': decay, 'momentum': momentum, 'epsilon': epsilon, 'centered': centered}}
        with st.expander('Graph'):
            graph_granularity = st.selectbox(label='Graph Granularity (By Epochs)',
                         options=[1, 5, 10, 20, 40, 100])
            # 更新session
            st.session_state['graph_granularity'] = graph_granularity
            # Train loss chart and test metric chart
            train_loss_chart = st.vega_lite_chart(data=None, spec={
                'height': 200,
                'mark': {'type': 'circle', 'tooltip': True},
                'encoding': {
                    'x': {'field': 'Epoch', 'type': 'quantitative'},
                    'y': {'field': 'Train Loss', 'type': 'quantitative'},
                }}, selection={
                "grid": {
                    "type": "interval", "bind": "scales"
                }}, use_container_width=True)
            test_metric_chart = st.vega_lite_chart(data=None, spec={
                'height': 200,
                'mark': {'type': 'circle', 'tooltip': True},
                'encoding': {
                    'x': {'field': 'Epoch', 'type': 'quantitative'},
                    'y': {'field': 'Test Metric', 'type': 'quantitative'},
                }}, selection={
                "grid": {
                    "type": "interval", "bind": "scales"
                }}, use_container_width=True)
    else:
        # 传统机器学习
        with st.expander('Set Parameters'):
            model_current = st.session_state['model']

            if model_current == 'LinearRegression':
                # LinearRegression()
                fit_intercept = st.checkbox(label='Fit Intercept', value=True, help='是否计算此模型的截距')
                st.session_state['kwargs'] = {'fit_intercept': fit_intercept}

            elif model_current == 'Lasso':
                # Lasso(alpha=0.025)
                alpha = st.number_input(label='Alpha', value=0.025, min_value=0.0001, format='%.4f',
                                help='乘以惩罚项的常数')
                st.session_state['kwargs'] = {'alpha': alpha}

            elif model_current == 'Ridge':
                # Ridge(alpha=0.002)
                alpha = st.number_input(label='Alpha', value=0.002, min_value=0.0001, format='%.4f',
                                help='乘以惩罚项的常数')
                st.session_state['kwargs'] = {'alpha': alpha}

            elif model_current == 'ElasticNet':
                # ElasticNet(alpha=0.02, l1_ratio=0.7)
                alpha = st.number_input(label='Alpha', value=0.02, min_value=0.0001, format='%.4f',
                                help='乘以惩罚项的常数')
                l1_ratio = st.number_input(label='L1 Ratio', value=0.7, min_value=0.0, max_value=1.0, format='%.4f',
                                help='Elastic-Net（弹性网）混合参数，取值范围0 <= l1_ratio <= 1')
                st.session_state['kwargs'] = {'alpha': alpha, 'l1_ratio': l1_ratio}

            elif model_current == 'BayesianRidge':
                # BayesianRidge(n_iter=300, alpha_1=1e-9, alpha_2=1e-9, lambda_1=1e-9, lambda_2=1e-9, alpha_init=0.2, compute_score=False)
                n_iter = st.number_input(label='N_iter', value=300, min_value=1, step=1,
                                help='最大迭代次数。应该大于或等于1')
                alpha_1 = st.number_input(label='Alpha_1', value=1e-9, format='%g',
                                help='高于alpha参数的Gamma分布的形状参数')
                alpha_2 = st.number_input(label='Alpha_2', value=1e-9, format='%g',
                                help='优先于alpha参数的Gamma分布的反比例参数（速率参数）')
                lambda_1 = st.number_input(label='Lambda_1', value=1e-9, format='%g',
                                help='高于lambda参数的Gamma分布的形状参数')
                lambda_2 = st.number_input(label='Lambda_2', value=1e-9, format='%g',
                                help='优先于lambda参数的Gamma分布的反比例参数（速率参数）')
                alpha_init = st.number_input(label='Alpha Init', value=0.2, step=0.001, format='%.3f',
                                help='alpha的初始值（噪声的精度）')
                compute_score = st.checkbox(label='ComputeScore', value=False,
                            help='如果为True，则计算模型每一步的目标函数')
                st.session_state['kwargs'] = {'n_iter': n_iter, 'alpha_1': alpha_1, 'alpha_2': alpha_2,
                                              'lambda_1': lambda_1, 'lambda_2': lambda_2, 'alpha_init': alpha_init,
                                              'compute_score': compute_score}

            elif model_current == 'KNN':
                # KNeighborsRegressor(n_neighbors=3)
                n_neighbors = st.number_input(label='N_neighbors', value=3, min_value=1, step=1,
                                help='用于kneighbors查询的临近点数')
                st.session_state['kwargs'] = {'n_neighbors': n_neighbors}

            elif model_current == 'SupportVector':
                # SVR(kernel='rbf')
                kernel = st.selectbox(label='Kernel',
                             options=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                             index=2,
                             help='指定算法中使用的内核类型')
                st.session_state['kwargs'] = {'kernel': kernel}

            elif model_current == 'DecisionTree':
                # DecisionTreeRegressor(max_depth=5)
                max_depth = st.number_input(label='Max Depth', value=5, min_value=0, step=1,
                                help='树的最大深度。如果为0也就是None，则将节点展开，直到所有叶子都是纯净的，或者直到所有叶子都包含少于min_samples_split个样本')
                st.session_state['kwargs'] = {'max_depth': max_depth}

            elif model_current == 'RandomForest':
                # RandomForestRegressor(criterion='squared_error', n_estimators=100, random_state=0)
                n_estimators = st.number_input(label='N_Estimators', value=100, min_value=1, step=1,
                               help='森林中树木的数量')
                max_depth = st.number_input(label='Max Depth', value=0, min_value=0, step=1,
                                help='树的最大深度。如果为0也就是None，则将节点展开，直到所有叶子都是纯净的，或者直到所有叶子都包含少于min_samples_split个样本')
                st.session_state['kwargs'] = {'n_estimators': n_estimators,
                                              'max_depth': max_depth}

            elif model_current == 'ExtraTree':
                # ExtraTreesRegressor(n_estimators=50)
                n_estimators = st.number_input(label='N_Estimators', value=50, min_value=1, step=1,
                                help='森林中树木的数量')
                max_depth = st.number_input(label='Max Depth', value=0, min_value=0, step=1,
                                help='树的最大深度。如果为0也就是None，则将节点展开，直到所有叶子都是纯净的，或者直到所有叶子都包含少于min_samples_split个样本')
                st.session_state['kwargs'] = {'n_estimators': n_estimators,
                                              'max_depth': max_depth}

            elif model_current == 'AdaBoost':
                # AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), n_estimators=100, random_state=None)
                max_depth = st.number_input(label='Max Depth', value=5, min_value=0, step=1,
                                help='默认的基学习器DecisionTreeRegressor树的最大深度。如果为0也就是None，则将节点展开')
                n_estimators = st.number_input(label='N_Estimators', value=100, min_value=1, step=1,
                                help='终止推进的估计器的最大数目。如果完全拟合，学习过程就会提前停止。')
                learning_rate = st.number_input(label='Learning Rate', value=1.0, min_value=0.0001, step=0.001, format='%.4f',
                                help='学习率,在每次提升迭代中应用于每个回归器的权重。')
                st.session_state['kwargs'] = {'n_estimators': n_estimators,
                                              'max_depth': max_depth,
                                              'learning_rate': learning_rate}

            elif model_current == 'GBDT':
                # GradientBoostingRegressor(n_estimators=50, max_depth=5, learning_rate=0.3)
                n_estimators = st.number_input(label='N_Estimators', value=50, min_value=1, step=1,
                                help='要执行的推进阶段的数量')
                max_depth = st.number_input(label='Max Depth', value=5, min_value=1, step=1,
                                help='单个回归估计量的最大深度')
                learning_rate = st.number_input(label='Learning Rate', value=0.3, min_value=0.0001, step=0.001, format='%.4f',
                                help='学习率通过learning_rate缩小每棵树的贡献')
                st.session_state['kwargs'] = {'n_estimators': n_estimators,
                                              'max_depth': max_depth,
                                              'learning_rate': learning_rate}

            elif model_current == 'Xgboost':
                # XGBRegressor(n_estimators=50, max_depth=5, learning_rate=0.3)
                n_estimators = st.number_input(label='N_Estimators', value=50, min_value=1, step=1,
                                help='梯度提升树的数量。 相当于提升轮数。')
                max_depth = st.number_input(label='Max Depth', value=5, min_value=1, step=1,
                                help='基础学习器的最大树深度')
                learning_rate = st.number_input(label='Learning Rate', value=0.3, min_value=0.0001, step=0.001, format='%.4f',
                                help='学习率')
                st.session_state['kwargs'] = {'n_estimators': n_estimators,
                                              'max_depth': max_depth,
                                              'learning_rate': learning_rate}

            elif model_current == 'LightGBM':
                # LGBMRegressor(n_estimators=100, max_depth=5, num_leaves=10, learning_rate=0.1)
                n_estimators = st.number_input(label='N_Estimators', value=100, min_value=1, step=1,
                                help='梯度提升树的数量')
                max_depth = st.number_input(label='Max Depth', value=5, step=1,
                                help='基础学习器的最大树深度，<=0 表示没有限制。')
                num_leaves = st.number_input(label='Num Leaves', value=10, min_value=1, step=1,
                                help='基础学习者的最大树叶')
                learning_rate = st.number_input(label='Learning Rate', value=0.1, min_value=0.0001, step=0.001, format='%.4f',
                                help='学习率')
                st.session_state['kwargs'] = {'n_estimators': n_estimators,
                                              'max_depth': max_depth,
                                              'num_leaves': num_leaves,
                                              'learning_rate': learning_rate}


st.title("🎈DEEP LEARNING MODEL & MACHINE LEARNING MODEL")
# Run button and Training Log Console
with st.container():
    # st.header('Running Log')
    if st.session_state['model'] in option_model_ml:
        st.info('ℹ️ Dashboard of Loss/Epoch/Batch/ is for deep learning models')
    c_run, c_metric, c_loss, c_epoch, c_batch, c_time = st.columns(6)
    with c_run:
        button_run = st.button('Run')
    with c_metric:
        st.markdown('**Metrics**')
        place_metric = st.empty()
        place_metric.markdown('0')
    with c_loss:
        if st.session_state['model'] in option_model_ml:
            st.markdown('**<font color=#DCDCDC>Loss</font>**', unsafe_allow_html=True)
            place_loss = st.empty()
            place_loss.markdown('<font color=#DCDCDC>0</font>', unsafe_allow_html=True)
        else:
            st.markdown('**Loss**')
            place_loss = st.empty()
            place_loss.markdown('0')
    with c_epoch:
        if st.session_state['model'] in option_model_ml:
            st.markdown('**<font color=#DCDCDC>Epoch</font>**', unsafe_allow_html=True)
            place_epoch = st.empty()
            # place_epoch.markdown(f'<font color=#DCDCDC>0 of {int(st.session_state["epochs"])}</font>',
            #                      unsafe_allow_html=True)
        else:
            st.markdown('**Epoch**')
            place_epoch = st.empty()
            place_epoch.markdown(f'0 of {int(st.session_state["epochs"])}')
    with c_batch:
        if st.session_state['model'] in option_model_ml:
            st.markdown('**<font color=#DCDCDC>Batch</font>**', unsafe_allow_html=True)
            place_batch = st.empty()
            # place_batch.markdown(
            #     f'<font color=#DCDCDC>0 of {int(np.ceil(int(st.session_state["batch_size_max_value"]) / int(st.session_state["batch_size"])))}</font>',
            # unsafe_allow_html=True)
        else:
            st.markdown('**Batch**')
            place_batch = st.empty()
            place_batch.markdown(f'0 of {int(np.ceil(int(st.session_state["batch_size_max_value"]) / int(st.session_state["batch_size"])))}')
    with c_time:
        st.markdown('**Training Time**')
        place_time = st.empty()
        place_time.markdown('0')

# # Train loss chart and test metric chart
# with st.container():
#     c_train_loss, c_test_metric = st.columns(2)
#     with c_train_loss:
#         st.markdown('### Train Loss')
#         train_loss_chart = st.vega_lite_chart(data=None, spec={
#                 'height': 300,
#                 'mark': {'type': 'circle', 'tooltip': True},
#                 'encoding': {
#                     'x': {'field': 'Epoch', 'type': 'quantitative'},
#                     'y': {'field': 'Train Loss', 'type': 'quantitative'},
#             }}, selection={
#                 "grid": {
#                     "type": "interval", "bind": "scales"
#             }}, use_container_width=True)
#     with c_test_metric:
#         st.markdown('### Test Metric')
#         test_metric_chart = st.vega_lite_chart(data=None, spec={
#                 'height': 300,
#                 'mark': {'type': 'circle', 'tooltip': True},
#                 'encoding': {
#                     'x': {'field': 'Epoch', 'type': 'quantitative'},
#                     'y': {'field': 'Test Metric', 'type': 'quantitative'},
#             }}, selection={
#                 "grid": {
#                   "type": "interval", "bind": "scales"
#             }}, use_container_width=True)

# Predicted Chart

container_predict_chart = st.container()
with container_predict_chart:
    st.markdown('### Test Predicted Chart')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("###### 均方误差MSE")
        place_col1 = st.empty()
        place_col1.metric("Mean Squared Error", "0")
    with col2:
        st.markdown("###### 平均绝对值误差MAE")
        place_col2 = st.empty()
        place_col2.metric("Mean Absolute Error", "0")
    with col3:
        st.markdown("###### 均方根误差RMSE")
        place_col3 = st.empty()
        place_col3.metric("Root Mean Square Error", "0")
    with col4:
        st.markdown("###### 决定系数R2 Score")
        place_col4 = st.empty()
        place_col4.metric("R2 Score", "0")
    # predicted chart
    test_predict_chart = st.vega_lite_chart(data=None, spec={
        "height": 300,
        "mark": {'type': "line", 'tooltip': True},
        "encoding": {
            "x": {"field": "Index", "type": "quantitative"},
            "y": {"field": "Value", "type": "quantitative"},
            "color": {"field": "Label",
                      "legend": {"title": None, "direction": "horizontal", "orient": "top"}
                      }
    }}, selection={
        "grid": {
          "type": "interval", "bind": "scales"
    }}, use_container_width=True)

# Console
# with st.container():
#     st.markdown('### Console')
#     st.write(st.session_state)
c_outline, c_content = st.columns((1.5, 2))
with c_outline:
    st.markdown("#### Outline")
    if st.session_state['model'] in option_model_ml:
        st.info('ℹ️ Outline is for deep learning model')
    model_select = st.session_state['model']
    if model_select == 'Conv1d':
        # layer group 1
        with st.expander("Conv1D"):
            st_conv1d(filters=256, key_name='conv1d_layer1')
        with st.expander("MaxPooling1D"):
            st_maxpool1d(poolsize=8, key_name='maxpool1d_layer1')
        # layer group 2
        with st.expander("Conv1D"):
            st_conv1d(filters=128, key_name='conv1d_layer2')
        with st.expander("MaxPooling1D"):
            st_maxpool1d(poolsize=8, key_name='maxpool1d_layer2')
        # layer group 3
        with st.expander("Conv1D"):
            st_conv1d(filters=64, key_name='conv1d_layer3')
        with st.expander("MaxPooling1D"):
            st_maxpool1d(poolsize=2, key_name='maxpool1d_layer3')
        # layer group 4
        with st.expander("Flatten"):
            st_flatten(info='ℹ️ Will flatten input')
        with st.expander("Dense"):
            st_dense(units=16, activation='none', key_name='conv1d_dense1')
        with st.expander("Dense"):
            st_dense(units=1, activation='none', key_name='conv1d_dense2')

    elif model_select == 'Conv2d':
        # intput layer
        with st.expander('InputLayer'):
            st.write('Input Shape')
            st.write(st.session_state['input_shape'])
        # layer group 1
        # exp_layer1_conv2d = st.expander("Conv2D")
        # st_conv2d(exp_layer1_conv2d, filters=16)
        with st.expander("Conv2D"):
            st_conv2d(filters=16, key_name='conv2d_layer1')
        with st.expander("MaxPooling2D"):
            st_maxpool2d(key_name='maxpool2d_layer1')
        # layer group 2
        with st.expander("Conv2D"):
            st_conv2d(filters=32, key_name='conv2d_layer2')
        with st.expander("MaxPooling2D"):
            st_maxpool2d(key_name='maxpool2d_layer2')
        # layer group 3
        with st.expander("Conv2D"):
            st_conv2d(filters=64, key_name='conv2d_layer3')
        with st.expander("MaxPooling2D"):
            st_maxpool2d(key_name='maxpool2d_layer3')
        # layer group 4
        with st.expander("Conv2D"):
            st_conv2d(filters=16, key_name='conv2d_layer4')
        with st.expander("MaxPooling2D"):
            st_maxpool2d(key_name='maxpool2d_layer4')
        # layer group 5
        with st.expander("Flatten"):
            st_flatten('ℹ️ Will flatten input. i.g.: [28, 28] => [784]')
        with st.expander("Dense"):
            st_dense(units=64, activation='Relu', key_name='conv2d_dense1')
        with st.expander("Dense"):
            st_dense(units=16, activation='Relu', key_name='conv2d_dense2')
        with st.expander("Dense"):
            st_dense(units=1, activation='none', key_name='conv2d_dense3')

    elif model_select == 'RNN':
        with st.expander("SimpleRNN"):
            st_rnn(units=40, activation='Tanh', return_seq=True, key_name='rnn_layer1')
        with st.expander("SimpleRNN"):
            st_rnn(units=40, activation='Tanh', return_seq=False, key_name='rnn_layer2')
        with st.expander("Dense"):
            st_dense(units=4, activation='none', key_name='rnn_dense1')
        with st.expander("Dense"):
            st_dense(units=1, activation='none', key_name='rnn_dense2')

    elif model_select == 'LSTM':
        with st.expander("LSTM"):
            st_lstm(units=40, activation='Tanh', recurrent_activation='Sigmoid',
                    return_seq=True, key_name='lstm_layer1')
        with st.expander("LSTM"):
            st_lstm(units=40, activation='Tanh', recurrent_activation='Sigmoid',
                    return_seq=False, key_name='lstm_layer2')
        with st.expander("Dense"):
            st_dense(units=4, activation='none', key_name='lstm_dense1')
        with st.expander("Dense"):
            st_dense(units=1, activation='none', key_name='lstm_dense2')

    elif model_select == 'BiLSTM':
        with st.expander("BidirectionalLSTM"):
            st_bilstm(units=40, activation='Tanh', recurrent_activation='Sigmoid',
                    return_seq=True, key_name='bilstm_layer1')
        with st.expander("BidirectionalLSTM"):
            st_bilstm(units=40, activation='Tanh', recurrent_activation='Sigmoid',
                    return_seq=False, key_name='bilstm_layer2')
        with st.expander("Dense"):
            st_dense(units=4, activation='none', key_name='bilstm_dense1')
        with st.expander("Dense"):
            st_dense(units=1, activation='none', key_name='bilstm_dense2')

    elif model_select == 'GRU':
        with st.expander("GRU"):
            st_gru(units=40, activation='Relu', recurrent_activation='Sigmoid',
                    return_seq=True, key_name='gru_layer1')
        with st.expander("GRU"):
            st_gru(units=40, activation='Relu', recurrent_activation='Sigmoid',
                      return_seq=False, key_name='gru_layer2')
        with st.expander("Dense"):
            st_dense(units=4, activation='none', key_name='gru_dense1')
        with st.expander("Dense"):
            st_dense(units=1, activation='none', key_name='gru_dense2')

with c_content:
    # st.markdown('#### Console')
    expander_log = st.expander("💬 Open log")
    with expander_log:
        # st.write("**Parameters:**")
        # st.write(st.session_state)
        # st.write("**Console:**")
        empty_log = st.empty()

# Button Clicked
if button_run:
    if st.session_state['dataset'] == 'PHM 2016 Challenge':

        with expander_log:
            st.write("**Parameters:**")
            st.write(st.session_state)
            st.write("**Console:**")
            st.write('Start: Hello World')

        model_type = st.session_state['model']
        if model_type in option_model_dp:
            # 深度学习模型
            # Step0:load data
            X_train, y_train, X_test, y_test = load_dataset(dataset_name=st.session_state['dataset'],
                                                            model_type=model_type)
            if st.session_state['model'] == 'Conv2d':
                # reshape数据 to update X_train and X_test
                wafer_number, max_batch_length, variable_number = X_train.shape
                wafer_number_test, max_batch_length, variable_number = X_test.shape
                X_train = X_train.reshape((wafer_number, max_batch_length, variable_number, 1))
                X_test = X_test.reshape((wafer_number_test, max_batch_length, variable_number, 1))
            with expander_log:
                st.write('Step 0: Load data done', X_train.shape, X_test.shape)

            # Step1:define net
            model = DeepLearningModel().init_model(model_type=st.session_state['model'])
            with expander_log:
                st.write('Step 1: Define net done', st.session_state['model'])

            # Step2:compile
            # e.g model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
            optimizer_str = st.session_state['optimizer']['optimizer']
            optimizer_kwargs = st.session_state['optimizer']['kwargs']
            lr = st.session_state['learning_rate']

            if optimizer_str == 'Sgd':
                opt = SGD(learning_rate=lr)
            elif optimizer_str == 'Momentum':
                opt = SGD(learning_rate=lr,
                          momentum=optimizer_kwargs['momentum'],
                          nesterov=optimizer_kwargs['use_nesterov'])
            elif optimizer_str == 'Adagrad':
                opt = Adagrad(learning_rate=lr,
                              initial_accumulator_value=optimizer_kwargs['initial_accumulator_value'])
            elif optimizer_str == 'Adadelta':
                opt = Adadelta(learning_rate=lr,
                               rho=optimizer_kwargs['rho'],
                               epsilon=optimizer_kwargs['epsilon'])
            elif optimizer_str == 'Adam':
                opt = Adam(learning_rate=lr,
                           beta_1=optimizer_kwargs['beta1'],
                           beta_2=optimizer_kwargs['beta2'],
                           epsilon=optimizer_kwargs['epsilon'])
            elif optimizer_str == 'Adamax':
                opt = Adamax(learning_rate=lr,
                             beta_1=optimizer_kwargs['beta1'],
                             beta_2=optimizer_kwargs['beta2'],
                             epsilon=optimizer_kwargs['epsilon'],
                             decay=optimizer_kwargs['decay'])
            elif optimizer_str == 'Rmsprop':
                opt = RMSprop(learning_rate=lr,
                              decay=optimizer_kwargs['decay'],
                              momentum=optimizer_kwargs['momentum'],
                              epsilon=optimizer_kwargs['epsilon'],
                              centered=optimizer_kwargs['centered'])
            model.compile(loss=st.session_state['loss_function'],
                          optimizer=opt,
                          metrics=['mse'])

            with expander_log:
                st.write('Step 2: Compile done')

            # Step3:fit
            n_epochs = int(st.session_state['epochs'])
            n_batch_size = int(st.session_state['batch_size'])
            n_step = int(np.ceil(int(st.session_state["batch_size_max_value"]) / n_batch_size))
            n_graph_granularity = int(st.session_state['graph_granularity'])
            # Funtion1:全部直接训练
            # model.fit(X_train, y_train, epochs=n_epochs, batch_size=n_batch_size)
            # st.write('model fit done')
            # Funtion2:分批次训练
            batch_data_list = batch_data_generate(X_train, y_train, batch_size=n_batch_size)  # 分批函数
            t1 = time.time()
            for i in range(0, n_epochs):
                # st.write(f'Epoch {i+1}/{n_epochs}')
                for j, (batch_x, batch_y) in enumerate(batch_data_list):
                    metrics = model.train_on_batch(batch_x, batch_y)
                    t2 = time.time()
                    # st.write(f'{j}: {metrics[0]} -- {t2-t1}')
                    with c_metric:
                        # st.write('Metrics')
                        place_metric.write(f'{metrics[0]:.4f}')
                    with c_loss:
                        # st.write('Loss')
                        place_loss.write(f'{metrics[0]:.4f}')
                    with c_epoch:
                        # st.write('Epoch')
                        place_epoch.write(f'{i+1} of {n_epochs}')
                    with c_batch:
                        # st.write('Batch')
                        place_batch.write(f'{j+1} of {n_step}')
                    with c_time:
                        # st.write('Training Time')
                        place_time.write(f'{(t2-t1)*1000:.0f}ms')
                # plot training loss and predicted metric by epoch graph_granularity
                if (i+1) % n_graph_granularity == 0:
                    y_test_pre_by_epoch = model.predict(X_test)
                    mse_by_epoch = mean_squared_error(y_test, y_test_pre_by_epoch)
                    df_train_loss_tmp = pd.DataFrame({'Epoch': (i+1), 'Train Loss': metrics[0]}, index={0})
                    df_test_metric_tmp = pd.DataFrame({'Epoch': (i+1), 'Test Metric': mse_by_epoch}, index={0})
                    # with c_train_loss:
                    train_loss_chart.add_rows(df_train_loss_tmp)
                    # with c_test_metric:
                    test_metric_chart.add_rows(df_test_metric_tmp)
            with expander_log:
                st.write('Step 3: Train Done')

            # Step4:predict test set
            y_test_pre = model.predict(X_test)
            with expander_log:
                st.write('Step 4: Predict test set done')

            # Step5:compute metrics
            mse = mean_squared_error(y_test, y_test_pre)
            rmse = np.sqrt(mean_squared_error(y_test, y_test_pre))
            mae = mean_absolute_error(y_test, y_test_pre)
            r2 = r2_score(y_test, y_test_pre)
            with expander_log:
                st.write('Compute metrics done')
                st.write(f'mse={mse:.4f}, rmse={rmse:.4f}, mae={mae:.4f}, r2={r2:.4f}')
                st.write('Step 5: Calculate metrics done')

            # Step6:plot predicted curve
            # Solution 1
            # fig = plt.figure(figsize=(15, 4))
            # plt.plot(np.arange(len(y_test)), y_test, color='royalblue', label='Ground Truth') #, linewidth=1.0, linestyle='-')
            # plt.plot(np.arange(len(y_test_pre)), y_test_pre, color='orange', label='Predicted') #, linewidth=1.0, linestyle='-', label='predict value')
            # # plt.title('Prediction Curve')
            # # plt.legend(loc='upper right')
            # plt.title('Test Set Prediction Results');plt.xlabel('No.');plt.ylabel('Value');plt.legend();plt.grid()
            # container_predict_chart.pyplot(fig)
            # Another
            # df_plot = pd.DataFrame(data=np.hstack([y_test, y_test_pre]),
            #                   columns=['Ground Truth', 'Predicted Value'])
            # st.line_chart(df_plot)
            # Solution 2
            # df_plot = pd.DataFrame(data=np.hstack([y_test, y_test_pre]),
            #                   columns=['Ground Truth', 'Predicted Value'])
            # Solution 3
            df_1 = pd.DataFrame(data=enumerate(y_test.reshape(1, -1)[0]),
                                columns=['Index', 'Value'])
            df_1['Label'] = 'Ground Truth'
            df_2 = pd.DataFrame(data=enumerate(y_test_pre.reshape(1, -1)[0]),
                                columns=['Index', 'Value'])
            df_2['Label'] = 'Predicted Value'
            df_plot = pd.concat([df_1, df_2]).reset_index(drop=True)
            with container_predict_chart:
                with col1:
                    place_col1.metric("Mean Absolute Error", f'{mse:.4f}')
                with col2:
                    place_col2.metric("Mean Absolute Error", f'{mae:.4f}')
                with col3:
                    place_col3.metric("Root Mean Square Error", f'{rmse:.4f}')
                with col4:
                    place_col4.metric("R2 Score", f'{r2:.4f}')
                test_predict_chart.add_rows(df_plot)
                # Debug
                # st.table(df_plot.head())
                # st.write(df_plot.shape)
                # st.write(df_3)
                # st.write(type(y_test[0][0]), type(y_test_pre[0][0]))
                # st.write(type(df_1.loc[0,'Index']), type(df_1.loc[0,'Value']))
                # st.write(type(df_2.loc[0, 'Index']), type(df_2.loc[0, 'Value']))
                # st.write(type(df_plot.loc[0, 'Index']), type(df_plot.loc[0, 'Value']))
                # st.write(type(df_plot.loc[0, 'Label']))

                # Print to container
                # st.markdown(f'**Mean Squared Error: {mse:.4f}**')
                # st.markdown(f'**Mean Absolute Error: {mae:.4f}**')
                # st.markdown(f'**Root Mean Squared Error: {rmse:.4f}**')
                # st.markdown(f'**R2 Score: {r2:.4f}**')
            with expander_log:
                st.write('Step 6: Predict Done')

        elif model_type in option_model_ml:
            # 机器学习模型
            # Step 0: load data
            train_x, train_y, test_x, test_y = load_dataset(dataset_name=st.session_state['dataset'],
                                                            model_type=model_type)
            with expander_log:
                st.write('Step 0: Load data done', train_x.shape, train_y.shape)

            # Step 1: init model
            ml_class = MachineLearingModel()
            model = ml_class.init_model(model_type=model_type, **st.session_state['kwargs'])
            with expander_log:
                st.write('Step 1: Init model done')
                st.markdown(f'<font color=#7CFC00>{model_type}</font>', unsafe_allow_html=True)

            # Step 2: fit and predict
            t1 = time.time()
            model, expected, predicted = ml_class.model_fit_predict(model, train_x, train_y.values.reshape(1, -1)[0], test_x, test_y)
            t2 = time.time()
            with c_time:
                # st.write('Training Time')
                place_time.write(f'{(t2 - t1) * 1000:.0f}ms')
            with expander_log:
                st.write('Step 2: fit and predict done')

            # Step 3: compute metrics
            mse = mean_squared_error(expected, predicted)
            rmse = np.sqrt(mean_squared_error(expected, predicted))
            mae = mean_absolute_error(expected, predicted)
            r2 = r2_score(expected, predicted)
            with c_metric:
                # st.write('Metrics')
                place_metric.write(f'{mse:.4f}')
            with expander_log:
                st.write('Step 3: Compute metrics done')
                st.write(f'mse={mse:.4f}, rmse={rmse:.4f}, mae={mae:.4f}, r2={r2:.4f}')

            # Step 4: plot curve
            df_1 = pd.DataFrame(data=enumerate(expected.values.reshape(1,-1)[0]), columns=['Index', 'Value'])
            df_1['Label'] = 'Ground Truth'
            df_2 = pd.DataFrame(data=enumerate(predicted), columns=['Index', 'Value'])
            df_2['Label'] = 'Predicted Value'
            df_plot = pd.concat([df_1, df_2]).reset_index(drop=True)
            with container_predict_chart:
                with col1:
                    place_col1.metric("Mean Absolute Error", f'{mse:.4f}')
                with col2:
                    place_col2.metric("Mean Absolute Error", f'{mae:.4f}')
                with col3:
                    place_col3.metric("Root Mean Square Error", f'{rmse:.4f}')
                with col4:
                    place_col4.metric("R2 Score", f'{r2:.4f}')
                test_predict_chart.add_rows(df_plot)
            with expander_log:
                st.write('Step 4: Plot curve done')
    else:
        with expander_log:
            st.info('ℹ️ TBC')







