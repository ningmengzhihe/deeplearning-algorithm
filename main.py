# System Tools
import time
# Data Analysis Tools
import numpy as np
import pandas as pd
# Machine Learning Tools
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Deep Learning Tools
from tensorflow.keras.optimizers import SGD, Adam, Adagrad, Adamax, Adadelta, RMSprop
# from keras.optimizer_v1 import SGD, Adam, Adagrad, Adamax, Adadelta, RMSprop
# Streamlit Tools
import streamlit as st
# UserDefine Class
from model import DeepLearningModel

# 初始化
config = {
    "dataset": "PHM 2016 Challenge",
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
option_conv2d_initializer = ['Constant', 'GlorotNormal', 'GlorotUniform', 'HeNormal', 'Identity', 'LeCunNormal', 'Ones',
                             'Orthogonal', 'RandomNormal', 'RandomUniform', 'TruncatedNormal', 'VarianceScaling', 'Zeros']
option_activation = ['none', 'Relu', 'Elu', 'Sigmoid', 'Softmax', 'Softplus', 'Tanh', 'Selu', 'Relu6']
# load data function
@st.cache
def load_dataset(dataset_name: str):
    '''
    dataset_name:str: st.selectbox
    :return:
    '''

    if dataset_name == 'PHM 2016 Challenge':
        # X_train = np.load("/Users/lihan/Workspace/data phm 2016/X_train_r_modeI_chamber4_mm.npy")
        # y_train = np.load("/Users/lihan/Workspace/data phm 2016/y_train_modeI_chamber4_mm.npy")
        # X_test = np.load("/Users/lihan/Workspace/data phm 2016/X_test_r_modeI_chamber4_mm.npy")
        # y_test = np.load("/Users/lihan/Workspace/data phm 2016/y_test_modeI_chamber4_mm.npy")
        X_train = np.load("./data/X_train_r_modeI_chamber4_mm.npy")
        y_train = np.load("./data/y_train_modeI_chamber4_mm.npy")
        X_test = np.load("./data/X_test_r_modeI_chamber4_mm.npy")
        y_test = np.load("./data/y_test_modeI_chamber4_mm.npy")

    return X_train, y_train, X_test, y_test

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
    st.number_input(label='Filters', value=filters, format='%d', key=f'num_filters_{key_name}')
    st.number_input(label='Kernel Size', value=2, format='%d', key=f'num_ker_size_{key_name}')
    st.number_input(label='Strides', value=1, format='%d', key=f'num_stride_{key_name}')
    st.selectbox(label='Padding',
                 options=['Valid', 'Same', 'Casual'],
                 index=0, key=f'sel_pad_{key_name}')
    st.info('ℹ️ Careful: There is no default activation on Conv1D layers')
    st.selectbox(label='Activation',
                 options=option_activation,
                 index=1, key=f'sel_active_{key_name}')
    st.checkbox(label='Use a Bias', value=True, key=f'check_bias_{key_name}')
    st.selectbox(label='Bias Initializer',
                 options=option_conv2d_initializer,
                 index=12, key=f'sel_bias_{key_name}')
    st.selectbox(label='Kernel Initializer',
                 options=option_conv2d_initializer,
                 index=2, key=f'sel_ker_init_{key_name}')

def st_conv2d(filters, key_name):
    st.number_input(label='Filters', value=filters, format='%d', key=f'num_filters_{key_name}')
    st.number_input(label='Kernel Size', value=2, format='%d', key=f'num_ker_size_{key_name}')
    st.number_input(label='Strides', value=1, format='%d', key=f'num_stride_{key_name}')
    st.selectbox(label='Padding',
                 options=['Valid', 'Same', 'Casual'],
                 index=1, key=f'sel_pad_{key_name}')
    st.info('ℹ️ Careful: There is no default activation on Conv2D layers')
    st.selectbox(label='Activation',
                 options=option_activation,
                 index=1, key=f'sel_active_{key_name}')
    st.checkbox(label='Use a Bias', value=True, key=f'check_bias_{key_name}')
    st.selectbox(label='Bias Initializer',
                 options=option_conv2d_initializer,
                 index=12, key=f'sel_bias_{key_name}')
    st.selectbox(label='Kernel Initializer',
                 options=option_conv2d_initializer,
                 index=2, key=f'sel_ker_init_{key_name}')


def st_maxpool1d(poolsize, key_name):
    st.number_input(label='Pool Size', value=poolsize, format='%d', key=f'num_pool_{key_name}')
    st.number_input(label='Strides', value=poolsize, format='%d', key=f'num_stride_{key_name}')
    st.selectbox(label='Padding',
                 options=['Valid', 'Same', 'Casual'],
                 index=0, key=f'sel_pad_{key_name}')


def st_maxpool2d(key_name):
    st.number_input(label='Pool Size', value=2, format='%d', key=f'num_pool_{key_name}')
    st.number_input(label='Strides', value=2, format='%d', key=f'num_stride_{key_name}')
    st.selectbox(label='Padding',
                 options=['Valid', 'Same', 'Casual'],
                 index=1, key=f'sel_pad_{key_name}')

def st_flatten(info):
    # st.info('ℹ️ Will flatten input. i.g.: [28, 28] => [784]')
    st.info(info)

def st_dense(units, activation, key_name):
    st.number_input(label='Units', value=units, key=f'num_units_{key_name}')
    st.selectbox(label='Activation',
                 options=option_activation,
                 index=option_activation.index(activation),
                 key=f'sel_active_{key_name}')
    st.selectbox(label='Kernel Initializer',
                 options=option_conv2d_initializer,
                 index=2, key=f'sel_ker_init_{key_name}')
    st.checkbox(label='Use a Bias', value=True, key=f'check_bias_{key_name}')
    st.selectbox(label='Bias Initializer',
                 options=option_conv2d_initializer,
                 index=12, key=f'sel_bias_{key_name}')

st.set_page_config(page_title="Deep Learning Model", page_icon="🎈", layout="wide")

st.session_state = config

with st.sidebar:
    # 菜单：读取数据/选择model/调节超参数/设置优化器
    with st.expander('Dataset'):
        # 选择数据集
        dataset_option = st.selectbox(
            label='How would you like to be contacted?',
            options=['PHM 2016 Challenge'])
        X_train, y_train, X_test, y_test = load_dataset(dataset_option)
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
        st.session_state['dataset'] = dataset_option

    with st.expander('Choose a model'):
        # 选择一种深度学习模型
        model_select = st.selectbox(
            label='Which model would you like to use?',
            options=['Conv1d', 'Conv2d', 'RNN', 'LSTM', 'BiLSTM', 'GRU'],
            index=0)
        # 更新session
        st.session_state['model'] = model_select

    with st.expander('Hyperparameters'):
        # 调节超参数
        batch_size = st.number_input(label='BatchSize:',
                                     min_value=1,
                                     max_value=st.session_state['batch_size_max_value'],
                                     step=1,
                                     value=st.session_state['batch_size'])
        epochs = st.number_input(label='Epochs',
                                 min_value=1,
                                 max_value=1000,
                                 step=10,
                                 value=st.session_state['epochs'])
        learning_rate = st.number_input(label='LearningRate',
                                        min_value=0.001,
                                        max_value=0.999,
                                        step=0.001,
                                        value=st.session_state['learning_rate'],
                                        format='%.3f')
        # 更新session
        st.session_state['batch_size'] = batch_size
        st.session_state['epochs'] = epochs
        st.session_state['learning_rate'] = learning_rate

    with st.expander('Optimizer'):
        # 设置损失函数
        loss_option = st.selectbox(label='Loss Function',
                                   options=['MeanSquaredError'],
                                   index=0)
        # 更新loss_function
        loss_dict = {'MeanSquaredError': 'mean_squared_error'}
        st.session_state['loss_function'] = loss_dict[loss_option]

        # 设置优化器
        optimizer_option = st.selectbox(label='Optimizer Function',
                                        options=['Sgd', 'Momentum', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Rmsprop'],
                                        index=4)
        if optimizer_option == 'Momentum':
            momentum = st.number_input(label='Momentum', min_value=0.0, value=0.0, step=0.01)
            use_nesterov = st.checkbox(label='useNesterov', value=False)
        elif optimizer_option == 'Adagrad':
            initial_accumulator_value = st.number_input(label='InitialAccumulatorValue',
                                                        value=0.1)
        elif optimizer_option == 'Adadelta':
            rho = st.number_input(label='Rho', value=0.95, min_value=0.0, step=0.01)
            epsilon = st.number_input(label='Epsilon', value=1e-7, min_value=0.0, step=0.005, format="%g")
        elif optimizer_option == 'Adam':
            beta1 = st.number_input(label='Beta1', value=0.9, min_value=0.001, max_value=0.99, step=0.05)
            beta2 = st.number_input(label='Beta2', value=0.999, min_value=0.001, max_value=0.999, step=0.005,
                                    format="%.3f")
            epsilon = st.number_input(label='Epsilon', value=1e-7, min_value=0.0, step=0.005, format="%g")
        elif optimizer_option == 'Adamax':
            beta1 = st.number_input(label='Beta1', value=0.9, min_value=0.001, max_value=0.999, step=0.05)
            beta2 = st.number_input(label='Beta2', value=0.999, min_value=0.001, max_value=0.999, step=0.005,
                                    format="%.3f")
            epsilon = st.number_input(label='Epsilon', value=1e-7, min_value=0.0, step=0.005, format="%g")
            decay = st.number_input(label='Decay', value=0.0, min_value=0.0, step=0.01)
        elif optimizer_option == 'Rmsprop':
            decay = st.number_input(label='Decay', value=0.0, min_value=0.0, step=0.01)
            momentum = st.number_input(label='Momentum', value=0.0, min_value=0.0, step=0.01)
            epsilon = st.number_input(label='Epsilon', value=1e-7, min_value=0.0, step=0.005, format="%g")
            centered = st.checkbox(label='centered', value=False)
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

st.title("🎈DEEP LEARNING MODEL")
# Run button and Training Log Console
with st.container():
    # st.header('Running Log')
    c_run, c_metric, c_loss, c_epoch, c_batch, c_time = st.columns(6)
    with c_run:
        button_run = st.button('Run')
    with c_metric:
        st.markdown('**Metrics**')
        place_metric = st.empty()
        place_metric.write('0')
    with c_loss:
        st.markdown('**Loss**')
        place_loss = st.empty()
        place_loss.write('0')
    with c_epoch:
        st.markdown('**Epoch**')
        place_epoch = st.empty()
        place_epoch.write(f'0 of {int(st.session_state["epochs"])}')
    with c_batch:
        st.markdown('**Batch**')
        place_batch = st.empty()
        place_batch.write(f'0 of {int(np.ceil(int(st.session_state["batch_size_max_value"]) / int(st.session_state["batch_size"])))}')
    with c_time:
        st.markdown('**Training Time**')
        place_time = st.empty()
        place_time.write('0')

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
        place_col1 = st.empty()
        place_col1.metric("MSE", "0")
    with col2:
        place_col2 = st.empty()
        place_col2.metric("MAE", "0")
    with col3:
        place_col3 = st.empty()
        place_col3.metric("RMSE", "0")
    with col4:
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
        with st.expander("Conv2d"):
            st_conv2d(filters=16, key_name='conv2d_layer1')
        with st.expander("MaxPooling2D"):
            st_maxpool2d(key_name='maxpool2d_layer1')
        # layer group 2
        with st.expander("Conv2d"):
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
            st.write("RNN")
        with st.expander("SimpleRNN"):
            st.write("RNN")
        with st.expander("Dense"):
            st.write("dense")
        with st.expander("Dense"):
            st.write("dense")

    elif model_select == 'LSTM':
        with st.expander("LSTM"):
            st.write("LSTM")
        with st.expander("LSTM"):
            st.write("LSTM")
        with st.expander("Dense"):
            st.write("dense")
        with st.expander("Dense"):
            st.write("dense")

    elif model_select == 'BiLSTM':
        with st.expander("BidirectionalLSTM"):
            st.write("BiLSTM")
        with st.expander("BidirectionalLSTM"):
            st.write("BiLSTM")
        with st.expander("Dense"):
            st.write("dense")
        with st.expander("Dense"):
            st.write("dense")

    elif model_select == 'GRU':
        with st.expander("GRU"):
            st.write("GRU")
        with st.expander("GRU"):
            st.write("GRU")
        with st.expander("Dense"):
            st.write("dense")
        with st.expander("Dense"):
            st.write("dense")

with c_content:
    st.markdown('#### Console')
    expander_log = st.expander("💬 Open log")
    with expander_log:
        st.write("**Parameters:**")
        st.write(st.session_state)
        st.write("**Console:**")

# Button Clicked
if button_run:
    with expander_log:
        st.write('Start: Hello World')
    # Step0:load data
    X_train, y_train, X_test, y_test = load_dataset(st.session_state['dataset'])
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

    # # Solution 1：配合tensorflow.keras.optimizers import SGD, Adam, Adagrad, Adamax, Adadelta, RMSprop使用
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
            place_col1.metric("MSE", f'{mse:.4f}')
        with col2:
            place_col2.metric("MAE", f'{mae:.4f}')
        with col3:
            place_col3.metric("RMSE", f'{rmse:.4f}')
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


