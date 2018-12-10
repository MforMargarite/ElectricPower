import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.layers.core import Dropout
import os
import pandas as pd
from matplotlib import pyplot
import numpy as np
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score
from keras.layers import Masking
from keras.layers import LeakyReLU
from keras.layers import Flatten
import keras.backend as K
import datetime


# 将小时数据映射到日期 增大故障类别的比例 计算roc_auc
def hour_to_day(series):
    result = []
    n = len(series)
    index = list(range(n))

    begin_date = pd.datetime.strptime(begin, '%Y-%m-%d %H:%M:%S')
    last = 0
    for time_split in index[24 - begin_date.hour: -1: 24]:
        result.append(np.max(series[last: time_split]))
        last = time_split
    if last != len(series)-1:
        result.append(np.max(series[last + 1: -1]))
    return result


def pre_process(data, cols):
    data.reset_index(inplace=True)
    for c in cols:
        cur_min = np.nanmin(data[c])
        cur_gap = np.nanmax(data[c]) - cur_min
        if cur_gap > 0:
            data[c] = data[c].apply(lambda x: (x-cur_min)/cur_gap if not pd.isnull(x) else -1)
        else:
            # 只有一个电源 情况
            data[c] = data[c].apply(lambda x: 0 if not pd.isnull(x) else -1)
    data_label = data.loc[:, 'result'].values
    data = data.loc[:, cols].values
    data = data.reshape(data.shape[0], 1, data.shape[1])
    return data[:-1], data_label[1:]


# 温度/电流/电压变化(变化率 方差)与故障率的关系 LSTM
# 归一化处理后(将值映射到0,1) 将缺失值置为-1
def get_lstm_data(path, r, filename):
    data = pd.read_csv(path + r + filename, encoding='utf-8')
    data.sort_values(by='TIME', ascending=True, inplace=True)
    n = data.shape[0]
    col = data.columns.tolist()
    col.remove('TIME')
    col.remove('result')
    train_sample = int(0.6 * n)+1
    # 归一化处理 将空值置为-1
    tr_X, tr_y = pre_process(data.loc[:train_sample, :], col)
    t_X, t_y = pre_process(data.loc[train_sample:, :], col)
    begin = data.loc[train_sample, 'TIME']
    return begin, tr_X, tr_y, t_X, t_y


def lstm_train(path, params, tr_X, tr_y, t_X, t_y):
    print(path)
    print(params)
    global model
    model = Sequential()
    n = len(params)
    for i in range(len(params)):
        if params[i] > 1:
            if i != len(params)-1 and params[i+1] > 1:
                model.add(LSTM(params[i], dropout=0.1 * (n-i), input_shape=(tr_X.shape[1], tr_X.shape[2]), return_sequences=True))
                model.add(LeakyReLU())
            else:
                model.add(LSTM(params[i], dropout=0.2, input_shape=(tr_X.shape[1], tr_X.shape[2])))
                break
        else:
            break
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    # 初始化roc_auc评价标准
    cb = [
        RocAucScore(valid_data=(t_X, t_y)),
        ModelCheckpoint(path + '/auc_{roc_auc:.2f}_lstm.h5', monitor='roc_auc', save_best_only=True, mode='max'),
        EarlyStopping(patience=25, mode='min')
    ]
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    # 训练模型
    history = model.fit(tr_X, tr_y, epochs=int(tr_X.shape[0] / 72), batch_size=72, validation_data=(t_X, t_y), verbose=2, shuffle=False, class_weight='auto', callbacks=cb)

    # 绘制训练和测试集误差曲线
    global fig
    pyplot.plot(history.history['loss'], label='train-loss')
    pyplot.plot(history.history['val_loss'], label='test-loss')
    pyplot.legend(loc='best')
    fig.savefig(path + '/' + str(params[0]) + '_loss.png', dpi=80, transparent=True)
    fig.clear()
    # 保留roc_auc最佳结果对应模型
    model_file = [f for f in os.listdir(path) if '.h5' in f]
    model_file.sort()
    score = model_file[-1].split("_")[1]
    return score


def file_name(parts):
    result = parts[1]
    for i in range(2, len(parts)):
        result += '_' + parts[i]
    return result


class RocAucScore(Callback):
    def __init__(self, valid_data):
        super(Callback, self).__init__()
        self.x_data = valid_data[0]
        self.y_true = valid_data[1]
        # self.y_true = hour_to_day(self.validation_data[1])

    def on_epoch_end(self, epoch, logs={}):
        # y_predict = hour_to_day(self.model.predict(self.validation_data[0]))
        y_predict = self.model.predict(self.x_data)
        if np.unique(self.y_true).shape[0] == 2:
            score = roc_auc_score(self.y_true, y_predict, average='weighted')
        else:
            score = 0.5
        logs['roc_auc'] = score
        print('roc_auc score on epoch ' + str(epoch + 1) + ':' + str(score) + '\n')


def gen_result_col(dirs):
    room_err_time = pd.read_csv(root + '/room_err_time.csv', encoding='utf-8')
    room_err_time = room_err_time.loc[:, ['PAR_ROOM','ALARM_EMS_TIME']]
    exists = []
    for r in room_err_time['PAR_ROOM'].unique().tolist():
        if os.path.exists(root + dirs + r + '/data_clean.csv'):
            data = pd.read_csv(root + dirs + r + '/data_clean.csv', encoding='utf-8')
            if 'result' not in data.columns.tolist():
                cur_time = room_err_time[room_err_time['PAR_ROOM'] == r]
                cur_time['ALARM_EMS_TIME'] = cur_time['ALARM_EMS_TIME'].apply(lambda x: datetime.datetime.strftime(datetime.datetime.fromtimestamp(x / 1000).replace(minute=0, second=0), '%Y-%m-%d %H:%M:%S'))
                err_time_list = cur_time['ALARM_EMS_TIME'].values.tolist()
                data['result'] = data['TIME'].apply(lambda x: 1 if x in err_time_list else 0)
                data.to_csv(root + dirs + r + '/data_clean.csv', header=True, index=False, encoding='utf-8')
            exists.append(r)
    return exists


def gen_lstm_by_attr(a):
    dirs = '/' + a + '/'
    room_list = gen_result_col(dirs)
    for par_room in room_list:
    #     r_files = os.listdir(root + dirs + par_room)
    #     for f in r_files:
    #         if '.h5' in f or '.png' in f:
    #             os.remove(root + dirs + par_room + '/' + f)
        begin, trainX, train_y, testX, test_y = get_lstm_data(root + dirs, par_room, '/data_clean.csv')
        lstm_train(root + dirs + par_room, [32,8,4], trainX, train_y, testX, test_y)


def multi_by_num(series, num):
    if 1 in series:
        index = series.index(1)
        for i in range(index + 1):
            series[i] = int(series[i] * num)
    return series


def gen_lstm(attrs, model):
    if not os.path.exists(root + '/combined/'):
        os.mkdir(root + '/combined/')

    room_list = []
    for a in attrs:
        room_list += gen_result_col('/' + a + '/')
    room_list = list(set(room_list))

    room_err_time = pd.read_csv(root + '/room_err_time.csv', encoding='utf-8')
    room_err_time = room_err_time.loc[:, ['PAR_ROOM', 'ALARM_EMS_TIME']]

    for r in room_list:
        if not os.path.exists(root + '/combined/' + r + '/'):
            os.mkdir(root + '/combined/' + r + '/')
        if not os.path.exists(root + '/combined/' + r + '/merge_data.csv'):
            cur_df = None
            for a in attrs:
                if os.path.exists(root + '/' + a + '/' + r + '/data_clean.csv'):
                    data = pd.read_csv(root + '/' + a + '/' + r + '/data_clean.csv', encoding='utf-8')
                    # 分离result列
                    cols = data.columns.tolist()
                    cols.remove('result')
                    data = data[cols]
                    for c in cols:
                        if c != 'TIME':
                            data.rename({c: a[:1].upper() + '_' + c}, axis=1, inplace=True)
                    if cur_df is None:
                        cur_df = data
                    else:
                        cur_df = pd.merge(cur_df, data, on='TIME', how='outer')
            cur_time = room_err_time[room_err_time['PAR_ROOM'] == r]
            cur_time['ALARM_EMS_TIME'] = cur_time['ALARM_EMS_TIME'].apply(lambda x: datetime.datetime.strftime(datetime.datetime.fromtimestamp(x / 1000).replace(minute=0, second=0),'%Y-%m-%d %H:%M:%S'))
            err_time_list = cur_time['ALARM_EMS_TIME'].values.tolist()
            cur_df['result'] = cur_df['TIME'].apply(lambda x: 1 if x in err_time_list else 0)
            if cur_df is not None:
                cur_df.to_csv(root + '/combined/' + r + '/merge_data.csv', header=True, index=False, encoding='utf-8')
                begin, trainX, train_y, testX, test_y = get_lstm_data(root + '/combined/', r, '/merge_data.csv')
                params = [4, 1, 1, 1]
                score = lstm_train(root + '/combined/' + r, params, trainX, train_y, testX, test_y)
                while float(score) < 0.7 and params[-1] == 1:
                    params = multi_by_num(params, 4)
                    score = lstm_train(root + '/combined/' + r, params, trainX, train_y, testX, test_y)
                # 保留最佳结果
                model_file = [f for f in os.listdir(root + '/combined/' + r) if '.h5' in f]
                model_file.sort()
                for model in model_file[:-1]:
                    os.remove(root + '/combined/' + r + '/' + model)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    attr = ['temperature', 'current', 'voltage']
    root = os.path.abspath(os.path.dirname(__file__))
    model = None
    fig = pyplot.figure()
    # for a in attr:
    #     gen_lstm_by_attr(a)
    gen_lstm(attr, model)
