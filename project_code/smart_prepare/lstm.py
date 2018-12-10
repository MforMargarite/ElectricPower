import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense
from keras.layers.core import Activation
# 增量学习
from keras.models import load_model
import os
from matplotlib import pyplot
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import datetime
import shutil
from sklearn.preprocessing import scale
from sklearn.preprocessing import MaxAbsScaler


def combine(col1, col2):
    result = []
    for i in range(len(col1)):
        result.append(str(int(col1[i])) + '_' + str(int(col2[i])))
    return result


# 获取各区县2016.01至2018.04每周各型号零件的需求量(实际备件需求从2016.09开始)
# 训练样本至2018.01
def gen_past_data():
    if os.path.exists(root + '/type_err_feature/'):
        shutil.rmtree(root + '/type_err_feature/')
    data = pd.read_csv(root + '/info/room_day_demand.csv', encoding='utf-8', usecols=['PRODUCER_CODE', 'DEV_TYPE', 'CENTER_CITY', 'ALARM_EMS_TIME_DAY', 'demand'])
    demand = data[['PRODUCER_CODE', 'DEV_TYPE', 'CENTER_CITY', 'ALARM_EMS_TIME_DAY', 'demand']].groupby(['PRODUCER_CODE', 'DEV_TYPE', 'CENTER_CITY', 'ALARM_EMS_TIME_DAY']).sum()
    demand.reset_index(inplace=True)
    demand['TYPE'] = combine(demand['PRODUCER_CODE'].values.tolist(), demand['DEV_TYPE'].values.tolist())
    centers = demand['CENTER_CITY'].unique().tolist()
    all_types = demand['TYPE'].unique().tolist()
    if not os.path.exists(root + '/type_err_feature/'):
        os.mkdir(root + '/type_err_feature/')
    for t in all_types:
        if not os.path.exists(root + '/type_err_feature/' + t + '/'):
            os.mkdir(root + '/type_err_feature/' + t + '/')
            os.mkdir(root + '/type_err_feature/' + t + '/center/')
        for c in centers:
            if not os.path.exists(root + '/type_err_feature/' + t + '/center/' + c + '/'):
                os.mkdir(root + '/type_err_feature/' + t + '/center/' + c + '/')
                cur = pd.DataFrame(demand[(demand['TYPE'] == t) & (demand['CENTER_CITY'] == c)])
                cur.drop('PRODUCER_CODE', axis=1, inplace=True)
                cur.drop('DEV_TYPE', axis=1, inplace=True)
                if cur.shape[0] > 0:
                    # 合并
                    all_time = pd.DataFrame()
                    all_time['ALARM_EMS_TIME_DAY'] = pd.date_range('2016-09-01', '2018-04-20', closed='left')
                    all_time['ALARM_EMS_TIME_DAY'] = all_time['ALARM_EMS_TIME_DAY'] .apply(lambda x: datetime.datetime.strftime(x,'%Y-%m-%d %H:%M:%S')[:-9])
                    all_time['TYPE'] = t
                    all_time['CENTER_CITY'] = c
                    all_time['demand'] = 0
                    cur.set_index(['TYPE', 'CENTER_CITY', 'ALARM_EMS_TIME_DAY'], inplace=True)
                    all_time.set_index(['TYPE', 'CENTER_CITY', 'ALARM_EMS_TIME_DAY'], inplace=True)
                    df = all_time.add(cur, fill_value=0)
                    df.reset_index(inplace=True)
                    # 按周整理
                    df['WEEK'] = df['ALARM_EMS_TIME_DAY'].apply(lambda x: str(datetime.datetime.strptime(x, '%Y-%m-%d').isocalendar()[0]) + '_' + str(datetime.datetime.strptime(x, '%Y-%m-%d').isocalendar()[1]).zfill(2))
                    df = df[['TYPE', 'CENTER_CITY', 'WEEK', 'demand']].groupby(['TYPE', 'CENTER_CITY', 'WEEK']).sum()
                    df.reset_index(inplace=True)
                    df['demand'] = df['demand'].apply(lambda x: round(x, 2))
                    df.drop(columns=['TYPE', 'CENTER_CITY'], axis=1, inplace=True)
                    df.to_csv(root + '/type_err_feature/' + t + '/center/' + c + '/demand_data.csv', encoding='utf-8', header=True, index=False)


def pre_process(data, steps):
    data_X, data_Y = [], []
    for i in range(len(data) - steps - 1):
        a = data.loc[i:(i + steps), 'demand'].values
        data_X.append(a)
        data_Y.append(data.loc[i + steps, 'demand'])
    return np.array(data_X), np.array(data_Y)


def plot_predict(data, steps, path, tr_predict, t_predict):
    tr_plot = np.empty_like(data)
    tr_plot[:, :] = np.nan
    tr_plot[steps:len(tr_predict) + steps, :] = tr_predict

    t_plot = np.empty_like(data)
    t_plot[:, :] = np.nan
    t_plot[len(tr_predict) + steps:len(data) - 1, :] = t_predict

    global fig
    plt.plot(data, alpha=1, color=colors[0], label='data')
    plt.plot(tr_plot, alpha=0.8, color=colors[1], label='train_predict')
    plt.plot(t_plot, alpha=0.8, color=colors[2], label='test_predict')
    plt.legend(loc="best")
    fig.savefig(path + '/predict.png', dpi=80, transparent=True)
    fig.clear()


def save_data(c, t, data, tr_predict, t_predict, rec_start, well_learnt):
    df = pd.DataFrame()
    df['city'] = [c]
    df['dev_type'] = [t]
    df['data'] = [data.reshape(1, -1).tolist()[0]]
    df['tr_predict'] = [tr_predict.reshape(1, -1).tolist()[0]]
    df['t_predict'] = [t_predict.reshape(1, -1).tolist()[0]]
    df['start_week'] = [rec_start]
    df['well_learnt'] = [well_learnt]
    return df


def model_train():
    all_types = os.listdir(root + '/type_err_feature/')
    result = pd.DataFrame()
    for t in all_types:
        all_cities = os.listdir(root + '/type_err_feature/' + t + '/center/')
        for c in all_cities:
            if os.path.exists(root + '/type_err_feature/' + t + '/center/' + c + '/demand_data.csv'):
                file = open(root + '/type_err_feature/' + t + '/center/' + c + '/demand_data.csv')
                data = pd.read_csv(file, encoding='utf-8')
                rec_start = data.loc[0, 'WEEK']

                col = ['demand']
                data = data.loc[:, col]
                if data['demand'].unique().tolist() != [0]:
                    scaler = MaxAbsScaler()
                    data = scaler.fit_transform(data)
                    data = pd.DataFrame(data[:, 0], columns=['demand'])
                    train_sample = int(0.7 * data.shape[0]) + 1

                    steps = 3
                    X, y = pre_process(data, steps)
                    X = scale(X, axis=1, with_std=False, with_mean=False)
                    tr_X, tr_y = X[:train_sample], y[:train_sample]
                    t_X, t_y = X[train_sample:], y[train_sample:]
                    tr_X, t_X = np.reshape(tr_X, (tr_X.shape[0], 1, tr_X.shape[1])), np.reshape(t_X, (t_X.shape[0], 1, t_X.shape[1]))

                    print(t, c)
                    path = root + '/type_err_feature/' + t + '/center/' + c
                    tr_predict, t_predict , well_learnt = lstm_train(path, tr_X, tr_y, t_X, t_y, scaler)
                    data = scaler.inverse_transform(data)
                    plot_predict(data, steps, path, tr_predict, t_predict)
                    result = result.append(save_data(c, t, data, tr_predict, t_predict, rec_start, well_learnt))
    result.to_csv(root + '/info/parts_prepare.csv', header=True, index=False, encoding='utf-8')


def lstm_train(path, tr_X, tr_y, t_X, t_y, scaler):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    model = Sequential()
    model.add(LSTM(32, input_shape=(tr_X.shape[1], tr_X.shape[2]), return_sequences=True))
    model.add(LSTM(16, input_shape=(tr_X.shape[1], tr_X.shape[2])))
    model.add(Dense(1))
    model.add(Activation('linear'))
    cb = [
        ModelCheckpoint(path + '/lstm.h5', monitor='val_loss', save_best_only=True, mode='min'),
        EarlyStopping(monitor='val_loss', patience=20, mode='max')
    ]
    model.compile(loss='mse', optimizer='adam')
    # 训练模型
    history = model.fit(tr_X, tr_y, epochs=tr_X.shape[0], batch_size=1, verbose=0, shuffle=False, callbacks=cb, validation_split=0.2)
    # 绘制训练和测试集误差曲线
    global fig
    pyplot.plot(history.history['loss'], label='train-loss')
    pyplot.plot(history.history['val_loss'], label='test-loss')
    pyplot.legend(loc='best')
    fig.savefig(path + '/loss.png', dpi=80, transparent=True)
    fig.clear()

    # 预测
    tr_predict = np.maximum(model.predict(tr_X), 0)
    t_predict = np.maximum(model.predict(t_X), 0)
    # 反归一化
    tr_y = scaler.inverse_transform([tr_y])
    t_y = scaler.inverse_transform([t_y])
    tr_predict = scaler.inverse_transform(tr_predict)
    t_predict = scaler.inverse_transform(t_predict)
    # 输出得分
    tr_score = math.sqrt(mean_squared_error(tr_y[0], tr_predict[:, 0]))
    print('Train Score: %.2f RMSE' % (tr_score))
    t_score = math.sqrt(mean_squared_error(t_y[0], t_predict[:, 0]))
    print('Test Score: %.2f RMSE' % (t_score))
    # 模型是否具有预测能力
    well_learnt = 0 if ((tr_score == 0.0 or t_score == 0.0) and abs(tr_score-t_score) > 0.1) or abs(tr_score-t_score) > 0.5 else 1
    return tr_predict, t_predict, well_learnt


def gen_month_demand():
    data = pd.read_csv(root + '/info/parts_prepare.csv', encoding='utf-8')
    week_to_month = [5, 4, 4, 5, 4, 4, 5, 4, 4, 5, 4, 4]    # 周到月的映射 考虑极端天气的需求量
    accu_week_to_month = [0, 5, 9, 13, 18, 22, 26, 31, 35, 39, 44, 48, 52]  # 累加 周到月的映射

    for index, row in data.iterrows():
        start_week = data.loc[index, 'start_week'].split('_')
        start_week = [int(start_week[0]), int(start_week[1])]
        offset = 3  # lstm算法的步长
        while start_week[1] + offset not in accu_week_to_month[:-1]:
            if start_week[1] + offset >= 52:
                start_week[0] += 1
                start_week[1] = (start_week[1] + offset) % 52
            else:
                offset += 1
        start_month = accu_week_to_month.index(start_week[1] + offset)
        true, predict = eval(row['data'])[offset + 1:-1], eval(row['tr_predict'])[offset-3+1:] + eval(row['t_predict'])
        true_by_month, predict_by_month = [], []
        num_before = 0
        for i in week_to_month[start_month:] + week_to_month:         # 2016.10至2017.12 共15个月
            true_by_month.append(np.sum(true[num_before: num_before + i]))
            predict_by_month.append(np.sum(predict[num_before: num_before + i]))
            num_before += i
        data.loc[index, 'true_by_month'] = str(true_by_month)
        data.loc[index, 'predict_by_month'] = str(predict_by_month)
        month_range = len(eval(data.loc[index, 'predict_by_month']))
        data.loc[index, 'start_month'] = str(start_week[0]) + '-' + str(accu_week_to_month.index(start_week[1] + offset)+1)
        data.loc[index, 'end_month'] = str(start_week[0] + int(month_range / 12)) + '-' + str(accu_week_to_month.index(start_week[1] + offset) + month_range % 12)
    data.to_csv(root + '/info/parts_prepare.csv', encoding='utf-8', header=True, index=False)


if __name__ == '__main__':
    root = os.path.abspath(os.path.dirname(__file__))
    colors = ['black', 'steelblue', 'indianred']
    fig = pyplot.figure()

    gen_past_data()
    model_train()
    # 周转月
    gen_month_demand()


