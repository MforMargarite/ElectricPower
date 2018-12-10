import pandas as pd
import numpy as np
import os
import datetime
import pymysql
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings


def time_transform(x):
    if x == '':
        return np.NaN
    possible_blank = x.find("月") + 1
    if x[possible_blank] == ' ':
        x = x[:possible_blank] + x[possible_blank + 1:]
    date_info = x.split(" ")[0].split("-")
    year = int(date_info[2])
    if year < datetime.datetime.now().year % 100:
        if year < 10:
            year = "0" + str(year)
        year = "20" + str(year)
    else:
        if year < 10:
            year = "0" + str(year)
        year = "19" + str(year)
    month = int(date_info[1].split("月")[0])
    if month < 10:
        month = "0" + str(month)
    return year + "-" + str(month) + "-" + str(date_info[0])


def get_dev_info():
    config = {
        'host': "172.16.135.6",
        'port': 3306,
        'user': 'root',
        'password': '10086',
        'db': 'jiangxi',
        'charset': 'utf8',
        'cursorclass': pymysql.cursors.DictCursor,
    }
    connection = pymysql.connect(**config)
    # 获取江西省 所有设备信息
    t_sql = "select t_rt_history_alarm.NE_OBJ_ID, t_ne.PAR_ROOM, t_ne.DEV_TYPE,t_pub_producer.PRODUCER_CODE, t_ne.BEG_TIME from t_rt_history_alarm,t_pub_producer, t_ne where t_pub_producer.OBJ_ID = t_ne.PRODUCER_ID and t_ne.OBJ_ID = t_rt_history_alarm.NE_OBJ_ID"
    df = pd.read_sql(t_sql, connection)
    df.drop_duplicates(inplace=True)
    df.fillna(-1, inplace=True)
    df.columns = ['NE_OBJ_ID', 'PAR_ROOM', 'DEV_TYPE', 'PRODUCER_CODE', 'BEG_TIME']
    df['BEG_TIME'] = df['BEG_TIME'].apply(time_transform)
    df['YEAR_USE'] = df['BEGIN_TIME'].apply(lambda x: int((datetime.datetime.now() - datetime.datetime.strptime(x, '%Y-%m-%d')).days / 365) if not pd.isnull(x) else -1)
    df.to_csv(root + '/dev_info.csv', header=True, index=False, encoding='utf-8')


def triphase_transform(c):
    if '_' in c:
        filtered = c.split("_")
        if '_A' in c or '_B' in c or '_C' in c:
            if len(filtered) == 3:
                return filtered[0] + "_" + filtered[2]
            else:
                return c
        else:
            return filtered[0]
    return c


# 根据设备厂家和型号 获取温度、电流、电压时序、使用年限、故障时序等属性 构造时序特征
def gen_feature_data_by_type():
    parent = root + '/type_err_feature/'
    if not os.path.exists(parent):
        os.mkdir(parent)
    dev_info = pd.read_csv(common + '/dev_info.csv', encoding='utf-8')
    series = pd.DataFrame(dev_info.groupby(['PRODUCER_CODE', 'DEV_TYPE']).size(), columns=['COUNT']).reset_index()
    room_err_time = pd.read_csv(common + '/room_err_time.csv', encoding='utf-8')
    room_err_time = room_err_time.loc[:, ['NE_OBJ_ID', 'ALARM_EMS_TIME']]
    room_err_time['ALARM_EMS_TIME'] = room_err_time['ALARM_EMS_TIME'].apply(lambda x: datetime.datetime.strftime(
        datetime.datetime.fromtimestamp(x / 1000).replace(minute=0, second=0), '%Y-%m-%d %H:%M:%S'))
    # 对于每种型号
    for index, row in series.iterrows():
        devs = dev_info[(dev_info['PRODUCER_CODE'] == row['PRODUCER_CODE']) & (dev_info['DEV_TYPE'] == row['DEV_TYPE'])].reset_index()
        devs.sort_values(by='NE_OBJ_ID', ascending=True, inplace=True)
        # 对于型号下的所有设备
        csv_data = pd.DataFrame()
        if not os.path.exists(parent + str(int(row['PRODUCER_CODE'])) + '_' + str(int(row['DEV_TYPE'])) + '/err_feature.csv'):
            for devindex, devrow in devs.iterrows():
                # dev_data = None
                # 判断设备是否发生故障
                cur_time = room_err_time[room_err_time['NE_OBJ_ID'] == devrow['NE_OBJ_ID']]
                err_time_list = cur_time['ALARM_EMS_TIME'].values.tolist()
                if len(err_time_list) > 0:
                    # 机房属性
                    # for a in attr:
                    #     if os.path.exists(root + '/' + a + '/' + str(devrow['PAR_ROOM']) + '/data_clean.csv'):
                    #         data = pd.read_csv(root + '/' + a + '/' + str(devrow['PAR_ROOM']) + '/data_clean.csv', encoding='utf-8')
                    #         # 分离result列
                    #         cols = data.columns.tolist()
                    #         cols.remove('result')
                    #         cols = [c for c in cols if 'VALUE' not in c]
                    #         data = data[cols]
                    #         for c in cols:
                    #             if c != 'TIME':
                    #                 data.rename({c: a[:1].upper() + '_'+c}, axis=1, inplace=True)
                    #
                    #         if dev_data is None:
                    #             dev_data = data
                    #         else:
                    #             dev_data = pd.merge(dev_data, data, on='TIME', how='outer')
                    #
                    # if dev_data is not None:

                    # 设备故障序列
                    dev_data = pd.DataFrame()
                    dev_data['TIME'] = pd.date_range('2016-01-03', '2018-04-20', freq='h')
                    dev_data['TIME'] = dev_data['TIME'].astype(str)
                    dev_data['result'] = dev_data['TIME'].apply(lambda x: 1 if x in err_time_list else 0)
                    # 设备使用年限
                    dev_data['YEAR_USE'] = dev_info[dev_info['NE_OBJ_ID'] == devrow['NE_OBJ_ID']]['YEAR_USE'].values[0]
                    # 构造时间特征
                    dev_data['TIME'] = dev_data['TIME'].apply(dateparse)
                    dev_data['dow'] = dev_data['TIME'].apply(lambda x: x.dayofweek)
                    dev_data['doy'] = dev_data['TIME'].apply(lambda x: x.dayofyear)
                    dev_data['month'] = dev_data['TIME'].apply(lambda x: x.month)
                    dev_data['hour'] = dev_data['TIME'].apply(lambda x: x.hour)
                    dev_data['result_before_1'] = [0] + list(dev_data['result'].values[:-1])
                    dev_data['result_before_7'] = [0 for i in range(7)] + list(dev_data['result'].values[:-7])
                    dev_data['result_before_30'] = [0 for i in range(30)] + list(dev_data['result'].values[:-30])
                    # 设备ID和机房ID
                    dev_data['NE_OBJ_ID'] = devrow['NE_OBJ_ID']
                    dev_data['PAR_ROOM'] = devrow['PAR_ROOM']
                    csv_data = csv_data.append(dev_data, ignore_index=True)

            if csv_data.shape[0] > 0:
                if not os.path.exists(parent + str(int(row['PRODUCER_CODE'])) + '_' + str(int(row['DEV_TYPE']))):
                    os.mkdir(parent + str(int(row['PRODUCER_CODE'])) + '_' + str(int(row['DEV_TYPE'])))

                # csv_data = labelize(csv_data, 'PAR_ROOM')
                csv_data.fillna(-1, inplace=True)
                # except_last = csv_data.shape[0] - 2
                # to_predict = pd.DataFrame()
                # to_predict[col] = csv_data.loc[:except_last, col]
                # to_predict['last_state'] = csv_data.loc[:except_last, 'result']
                # to_predict['result'] = csv_data.loc[1:, 'result'].values
                csv_data.to_csv(parent + str(int(row['PRODUCER_CODE'])) + '_' + str(int(row['DEV_TYPE'])) + '/err_feature.csv', header=True, index=False, encoding='utf-8')


def get_predict_model():
    parent = root + '/type_err_feature/'
    types = os.listdir(parent)
    for t in types:
        print(t)
        if not os.path.exists(parent + t + '/gdbt.model'):
        #if True:
            data = pd.read_csv(parent + t + '/err_feature.csv', encoding='utf-8', usecols=['result','YEAR_USE','dow','doy','month','hour','result_before_1','result_before_7'])
            n_estimators = range(40, 81, 10)
            min_sample_split = range(20, 81, 5)
            if data.shape[0] > 1000000:
                n_estimators = range(250, 501, 50)
                min_sample_split = range(120, 241, 30)
            elif data.shape[0] > 500000:
                n_estimators = range(100, 351, 50)
                min_sample_split = range(60, 141, 20)
            col = [c for c in data.columns.tolist() if c not in ['TIME', 'PAR_ROOM', 'NE_OBJ_ID', 'result']]
            trainX, testX, train_y, test_y = train_test_split(data[col], data['result'], test_size=0.3, random_state=80, stratify=data['result'])
            param_grid = {
                'n_estimators': n_estimators,
                'min_samples_split': min_sample_split
            }
            estimator = GradientBoostingClassifier(random_state=80, max_depth=5, learning_rate=0.005)
            cv = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=80)
            gbm = GridSearchCV(estimator=estimator, param_grid=param_grid, refit=True, n_jobs=-1, return_train_score=True,
                               scoring='roc_auc', cv=cv)
            # 训练和输出
            if train_y.values.tolist().count(1) >= 3:
                gbm.fit(trainX, train_y)
                # 预测
                y_pred = gbm.predict(testX)
                y_predprob = gbm.predict_proba(testX)[:, 1]
                print("型号 " + t + " roc_auc得分为：" + str(metrics.roc_auc_score(test_y, y_predprob, average='weighted')))
                # 输出结果至csv
                clf = gbm.best_estimator_
                result = pd.DataFrame()
                result["BEST_PARAMS"] = [str(gbm.best_params_)]
                result["FEATURE_RANK"] = [str(clf.feature_importances_)]
                result["TRAIN_ROC_AUC"] = [str(gbm.best_score_)]
                result["ACCURACY"] = [metrics.accuracy_score(test_y.values, y_pred)]
                result["ROC_AUC"] = [metrics.roc_auc_score(test_y, y_predprob, average='weighted')]
                result['y_pred'] = [y_pred.tolist()]
                result['y_true'] = [test_y.tolist()]
                result.to_csv(parent + t + '/gdbt_result.csv', header=True, index=False, encoding='utf-8')
                joblib.dump(clf, parent + t + '/gdbt.model')
            else:
                print(t + "型号样本故障数据过少，无法训练模型")


def gen_col_label(col):
    if not os.path.exists(root + '/label/' + col + '_label.model'):
        encoder = LabelEncoder()
        par_rooms = pd.read_csv(root + '/room_device_mapping.csv', encoding='utf-8')
        encoder.fit(par_rooms['PAR_ROOM'])
        joblib.dump(encoder, root + '/room_label.model')
    else:
        encoder = joblib.load(root + '/label/' + col + '_label.model')
    return encoder


# 构造时间特征 day_of_month, dat_of_week, hour_of_day, par_room, 12h变化情况
if __name__ == '__main__':
    common = os.path.abspath(os.path.dirname(__file__)) + '/trend'
    root = os.path.abspath(os.path.dirname(__file__))
    attr = ['temperature', 'current', 'voltage']
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')

    gen_feature_data_by_type()
    get_predict_model()
