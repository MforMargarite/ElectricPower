import pandas as pd
import numpy as np
import os
import datetime
import pymysql
import math
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import json
from sklearn.utils import shuffle


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
    if not os.path.exists(root + '/info/dev_info.csv'):
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
        # 获取设备信息(除温湿度、电源等)
        t_sql = "select DISTINCT t_ne.OBJ_ID, PAR_ROOM, DEV_TYPE, PRODUCER_CODE, BEG_TIME from t_pub_producer, t_ne where t_pub_producer.OBJ_ID = t_ne.PRODUCER_ID and t_ne.FULL_NAME not like '%电源%' and t_ne.FULL_NAME not like '%温湿度%'"
        df = pd.read_sql(t_sql, connection)
        df.drop_duplicates(inplace=True)
        df.fillna(-1, inplace=True)
        df.columns = ['NE_OBJ_ID', 'PAR_ROOM', 'DEV_TYPE', 'PRODUCER_CODE', 'BEG_TIME']
        df['BEG_TIME'] = df['BEG_TIME'].apply(time_transform)
        df['YEAR_USE'] = df['BEG_TIME'].apply(lambda x: int((datetime.datetime.now() - datetime.datetime.strptime(x, '%Y-%m-%d')).days / 365) if not pd.isnull(x) else -1)
        # 去除没有型号信息的设备
        df = df[(df['PRODUCER_CODE'].notna()) & (df['DEV_TYPE'].notna())]
        df.to_csv(root + '/info/dev_info.csv', header=True, index=False, encoding='utf-8')


def gen_abnormal_label(data, col, start, end, threshold):
    if col in data.columns.tolist():
        data = data[col].values.tolist()
        result = [np.NaN for i in range(start)]
        for i in range(start, len(data)):
            num = 0
            for j in range(i-start, i-end+1):
                if not math.isnan(data[j]) and (data[j] > threshold[1] or data[j] < threshold[0]):
                    num += 1
            result.append(num)
    else:
        result = [0 for i in range(data.shape[0])]
    return result


# 按设备厂家和型号 获取温度、电流、电压时序、使用年限、故障时序等属性 构造时序特征
def gen_feature_by_type():
    parent = root + '/type_err_feature/'
    if not os.path.exists(parent):
        os.mkdir(parent)
    dev_info = pd.read_csv(root + '/info/dev_info.csv', encoding='utf-8')
    series = pd.DataFrame(dev_info.groupby(['PRODUCER_CODE', 'DEV_TYPE']).size(), columns=['COUNT']).reset_index()
    room_err_time = pd.read_csv(common + '/room_err_time.csv', encoding='utf-8')

    room_err_time = room_err_time.loc[:, ['NE_OBJ_ID', 'ALARM_EMS_TIME']]
    room_err_time['ALARM_EMS_TIME'] = room_err_time['ALARM_EMS_TIME'].apply(lambda x: datetime.datetime.strftime(
        datetime.datetime.fromtimestamp(x / 1000).replace(minute=0, second=0), '%Y-%m-%d %H:%M:%S'))
    v_normal_range = json.load(open(common + '/data/voltage/normal_feature.txt', 'r'))
    t_normal_range = json.load(open(common + '/data/temperature/normal_feature.txt', 'r'))
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
                    if os.path.exists(common + '/data/voltage/' + str(devrow['PAR_ROOM']) + '/data_clean.csv'):
                        dev_data = pd.read_csv(common + '/data/voltage/' + str(devrow['PAR_ROOM']) + '/data_clean.csv', encoding='utf-8')
                        # 分离result列
                        cols = dev_data.columns.tolist()
                        cols.remove('result')
                        cols = ['TIME'] + [c for c in cols if 'VALUE' not in c and 'DIFF_RATE' not in c and '_0_' in c]
                        dev_data = dev_data[cols]
                        for c in cols:
                            if c != 'TIME':
                                splits = c.split('_')
                                dev_data.rename({c: 'V_'+splits[0]+'_'+splits[2]}, axis=1, inplace=True)
                        if os.path.exists(common + '/data/voltage/' + str(devrow['PAR_ROOM']) + '/data_clean.csv'):
                            t_dev_data = pd.read_csv(common + '/data/temperature/' + str(devrow['PAR_ROOM']) + '/data_clean.csv', encoding='utf-8')
                            # 分离result列
                            cols = t_dev_data.columns.tolist()
                            cols.remove('result')
                            cols = ['TIME'] + [c for c in cols if
                                               'VALUE' not in c and 'DIFF_RATE' not in c and '_0_' in c]
                            t_dev_data = t_dev_data[cols]
                            for c in cols:
                                if c != 'TIME':
                                    t_dev_data.rename({c: 'T_' + c}, axis=1, inplace=True)
                            dev_data = pd.merge(dev_data, t_dev_data, on='TIME')
                            # 设备故障序列
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
                            dev_data['result_before_6'] = [np.NaN for i in range(6)] + list(dev_data['result'].values[:-6])
                            dev_data['result_before_12'] = [np.NaN for i in range(12)] + list(dev_data['result'].values[:-12])
                            dev_data['result_before_24'] = [np.NaN for i in range(24)] + list(dev_data['result'].values[:-24])
                            dev_data['exist_abnormal_24_T_DIFF'] = gen_abnormal_label(dev_data, 'T_DIFF', 24, 12,t_normal_range['DIFF'])
                            dev_data['exist_abnormal_24_T_VAR'] = gen_abnormal_label(dev_data, 'T_VAR', 24, 12,t_normal_range['VAR'])
                            dev_data['exist_abnormal_12_T_DIFF'] = gen_abnormal_label(dev_data, 'T_DIFF', 12, 6,t_normal_range['DIFF'])
                            dev_data['exist_abnormal_12_T_VAR'] = gen_abnormal_label(dev_data, 'T_VAR', 12, 6,t_normal_range['VAR'])
                            dev_data['exist_abnormal_6_T_DIFF'] = gen_abnormal_label(dev_data, 'T_DIFF', 6, 1,t_normal_range['DIFF'])
                            dev_data['exist_abnormal_6_T_VAR'] = gen_abnormal_label(dev_data, 'T_VAR', 6, 1, t_normal_range['VAR'])

                            # 设备ID和机房ID
                            dev_data.dropna(inplace=True)
                            dev_data['NE_OBJ_ID'] = devrow['NE_OBJ_ID']
                            dev_data['PAR_ROOM'] = devrow['PAR_ROOM']
                            csv_data = csv_data.append(dev_data, ignore_index=True)

            if csv_data.shape[0] > 0:
                if not os.path.exists(parent + str(int(row['PRODUCER_CODE'])) + '_' + str(int(row['DEV_TYPE']))):
                    os.mkdir(parent + str(int(row['PRODUCER_CODE'])) + '_' + str(int(row['DEV_TYPE'])))

                csv_data.to_csv(parent + str(int(row['PRODUCER_CODE'])) + '_' + str(int(row['DEV_TYPE'])) + '/err_feature.csv', header=True, index=False, encoding='utf-8')


def gen_predict_model():
    parent = root + '/type_err_feature/'
    types = os.listdir(parent)
    for t in types:
        print(t)
        if not os.path.exists(parent + t + '/lightgbm.model'):
        # if True:
            data = pd.read_csv(parent + t + '/err_feature.csv', encoding='utf-8',usecols=['TIME', 'result', 'YEAR_USE', 'dow', 'doy', 'month', 'hour', 'result_before_6','result_before_12', 'result_before_24',
                                                                                        'exist_abnormal_24_T_DIFF','exist_abnormal_24_T_VAR', 'exist_abnormal_12_T_DIFF','exist_abnormal_12_T_VAR', 'exist_abnormal_6_T_DIFF', 'exist_abnormal_6_T_VAR' ])
            # 训练样本的数据 取2017.10.1 00:00:00前
            data['TIME'] = data['TIME'].apply(lambda x: dateparse(x))
            data = data[data['TIME'] <= dateparse('2017-10-01 00:00:00')]
            data.drop(columns='TIME', axis=1, inplace=True)

            try:
                col = [c for c in data.columns.tolist() if c != 'result']
                data = shuffle(data)
                trainX, testX, train_y, test_y = train_test_split(data[col], data['result'], test_size=0.3,random_state=80, stratify=data['result'])
                n_estimators = range(10, 121, 10)
                min_split = range(10, 151, 10)
                num_leaves = [63, 127, 255]
                param_grid = {
                    'min_child_samples': min_split,
                    'n_estimators': n_estimators,
                    'num_leaves': num_leaves,
                }
                # 增量学习
                # gbm = lightgbm.train(param_grid, trainX, num_boost_round=1000,valid_sets=train_y,init_model=None, feature_name=col,early_stopping_rounds=10,verbose_eval=False,keep_training_booster=True)
                estimator = LGBMClassifier(learning_rate=0.001, random_state=80, metrics='auc', objective='binary', is_unbalance=True, colsample_bytree=0.8)
                # 训练模型
                gbm = GridSearchCV(estimator=estimator, param_grid=param_grid, refit=True, scoring='roc_auc', cv=3)
                # 训练和输出
                if train_y.values.tolist().count(1) >= 3 and test_y.values.tolist().count(1) >= 3:
                    gbm.fit(trainX, train_y)
                    # 预测
                    y_pred = gbm.predict(testX)
                    print("型号 " + t + " roc auc得分为：" + str(metrics.roc_auc_score(test_y, y_pred)))
                    # 输出结果至csv
                    clf = gbm.best_estimator_
                    result = pd.DataFrame()
                    result["BEST_PARAMS"] = [str(clf)]
                    result["FEATURE_RANK"] = [str(clf.feature_importances_)]
                    result["TRAIN_SCORE"] = [str(gbm.best_score_)]
                    result["ACCURACY"] = [metrics.accuracy_score(test_y.values, y_pred)]
                    result["roc_auc_score"] = [metrics.roc_auc_score(test_y, y_pred)]
                    result['y_pred'] = [y_pred.tolist()]
                    result['y_true'] = [test_y.tolist()]
                    result.to_csv(parent + t + '/gbm_result.csv', header=True, index=False, encoding='utf-8')
                    joblib.dump(clf, parent + t + '/lightgbm.model')
            except ValueError as e:
                print(e)


# 构造时间特征 day_of_month, dat_of_week, hour_of_day, par_room, 12h变化情况
if __name__ == '__main__':
    common = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    root = os.path.abspath(os.path.dirname(__file__))
    attr = ['temperature', 'current', 'voltage']
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')

    # get_dev_info()
    gen_feature_by_type()
    gen_predict_model()
