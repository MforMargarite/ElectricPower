import pandas as pd
import numpy as np
import os
import datetime
import time
from sklearn.externals import joblib
import json
import math


def get_time_before(x, hour):
    hour *= -1
    if type(x) == str:
        x = datetime.datetime.strptime(x, '%Y-%m-%d')
    before = x + datetime.timedelta(hours=hour)
    t = before.timetuple()
    return time.strftime('%Y-%m-%d %H:00:00', t)


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


# 设备-机房-地区
# 参数: time: tuple (begin, end)
def predict_parts_change(begin, finish):
    if os.path.exists(root + '/info/predict_dev_err.csv'):
        os.remove(root + '/info/predict_dev_err.csv')
    all_dev = pd.read_csv(root + '/info/dev_info.csv', encoding='utf-8')
    if 'CITY' not in all_dev.columns.tolist():
        all_dev = gen_demand_by_city(all_dev)
        all_dev.to_csv(root + '/info/dev_info.csv', encoding='utf-8', index=False, header=True)

    # 获取所有城市
    cities = pd.read_table(root + '/data/jiangxi_city.txt', header=None, encoding='utf-8', delim_whitespace=True).values[0]
    cities = list(filter(lambda x: x.endswith('市') and x.count('市') == 1, cities))
    err_time = pd.read_csv(root + '/info/room_err_time.csv', encoding='utf-8', usecols=['NE_OBJ_ID', 'ALARM_EMS_TIME'])
    err_time['ALARM_EMS_TIME'] = err_time['ALARM_EMS_TIME'].apply(lambda x: datetime.datetime.strftime(datetime.datetime.fromtimestamp(x / 1000).replace(minute=0, second=0), '%Y-%m-%d %H:%M:%S'))
    v_normal_range = json.load(open(common + '/data/voltage/normal_feature.txt', 'r'))
    t_normal_range = json.load(open(common + '/data/temperature/normal_feature.txt', 'r'))

    for city in cities:
        city_dev = all_dev[all_dev['CITY'] == city[:-1]]
        city_dev.dropna(inplace=True)

        feature = pd.DataFrame()
        # 对每一个设备
        for index, row in city_dev.iterrows():
            dev_type = str(row['PRODUCER_CODE']) + '_' + str(int(row['DEV_TYPE']))
            if os.path.exists(root + '/type_err_feature/' + dev_type + '/lightgbm.model') and os.path.exists(common + '/data/voltage/' + str(row['PAR_ROOM']) + '/data_clean.csv'):
                cur = pd.read_csv(common + '/data/voltage/' + str(row['PAR_ROOM']) + '/data_clean.csv',encoding='utf-8')
                cols = cur.columns.tolist()
                cols.remove('result')
                cols = [c for c in cols if 'VALUE' not in c and 'DIFF_RATE' not in c]
                cur = cur[cols]
                cur = cur[(cur['TIME']>=start) & (cur['TIME']<=end)]
                cur.reset_index(inplace=True)
                for c in cols:
                    if c != 'TIME':
                        cur.rename({c: 'T_' + c}, axis=1, inplace=True)
                cur['TIME'] = cur['TIME'].apply(lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
                cur['YEAR_USE'] = row['YEAR_USE']
                cur['dow'] = cur['TIME'].apply(lambda x: x.dayofweek)
                cur['doy'] = cur['TIME'].apply(lambda x: x.dayofyear)
                cur['month'] = cur['TIME'].apply(lambda x: x.month)
                cur['hour'] = cur['TIME'].apply(lambda x: x.hour)
                # 获取前一天故障信息 按时间顺序预测
                err_time_list = err_time[err_time['NE_OBJ_ID'] == row['NE_OBJ_ID']].loc[:, 'ALARM_EMS_TIME'].values.tolist()
                cur['result'] = cur['TIME'].apply(lambda x: 1 if pd.datetime.strftime(x, '%Y-%m-%d %H:%M:%S') in err_time_list else 0)
                cur.loc[:5, 'result_before_6'] = cur.loc[:5, 'TIME'].apply(lambda x: 1 if get_time_before(x, 6) in err_time_list else 0)
                cur.loc[:11, 'result_before_12'] = cur.loc[:11, 'TIME'].apply(lambda x: 1 if get_time_before(x, 12) in err_time_list else 0)
                cur.loc[:23, 'result_before_24'] = cur.loc[:23, 'TIME'].apply(lambda x: 1 if get_time_before(x, 24) in err_time_list else 0)
                cur['exist_abnormal_24_T_DIFF'] = gen_abnormal_label(cur, 'T_DIFF', 24, 12, t_normal_range['DIFF'])
                cur['exist_abnormal_24_T_VAR'] = gen_abnormal_label(cur, 'T_VAR', 24, 12, t_normal_range['VAR'])
                cur['exist_abnormal_12_T_DIFF'] = gen_abnormal_label(cur, 'T_DIFF', 12, 6, t_normal_range['DIFF'])
                cur['exist_abnormal_12_T_VAR'] = gen_abnormal_label(cur, 'T_VAR', 12, 6, t_normal_range['VAR'])
                cur['exist_abnormal_6_T_DIFF'] = gen_abnormal_label(cur, 'T_DIFF', 6, 1, t_normal_range['DIFF'])
                cur['exist_abnormal_6_T_VAR'] = gen_abnormal_label(cur, 'T_VAR', 6, 1, t_normal_range['VAR'])
                cur['PAR_ROOM'] = row['PAR_ROOM']
                cursor = 1
                col = ['YEAR_USE', 'dow', 'doy', 'month', 'hour', 'result_before_6','result_before_12', 'result_before_24',
                       'exist_abnormal_24_T_DIFF', 'exist_abnormal_24_T_VAR', 'exist_abnormal_12_T_DIFF',
                       'exist_abnormal_12_T_VAR', 'exist_abnormal_6_T_DIFF', 'exist_abnormal_6_T_VAR'
                       ]
                #'exist_abnormal_24_V_DIFF_A','exist_abnormal_24_V_VAR_A','exist_abnormal_12_V_DIFF_A',
                       # 'exist_abnormal_12_V_VAR_A','exist_abnormal_6_V_DIFF_A', 'exist_abnormal_6_V_VAR_A','exist_abnormal_24_V_DIFF_B','exist_abnormal_24_V_VAR_B','exist_abnormal_12_V_DIFF_B',
                       # 'exist_abnormal_12_V_VAR_B','exist_abnormal_6_V_DIFF_B', 'exist_abnormal_6_V_VAR_B','exist_abnormal_24_V_DIFF_C','exist_abnormal_24_V_VAR_C','exist_abnormal_12_V_DIFF_C',
                       # 'exist_abnormal_12_V_VAR_C', 'exist_abnormal_6_V_DIFF_C','exist_abnormal_6_V_VAR_C',

                rf = joblib.load(root + '/type_err_feature/' + dev_type + '/lightgbm.model')
                for dev_i in range(cur.shape[0]):
                    predict_feature = cur.loc[dev_i, col].values.reshape(1, -1)
                    result = rf.predict(predict_feature)

                    cur.loc[dev_i, 'result'] = result[0]
                    if cursor + 5 < cur.shape[0]:
                        cur.loc[cursor + 5, 'result_before_6'] = result[0]
                    if cursor + 11 < cur.shape[0]:
                        cur.loc[cursor + 11, 'result_before_12'] = result[0]
                    if cursor + 23 < cur.shape[0]:
                        cur.loc[cursor + 23, 'result_before_24'] = result[0]
                    cursor += 1

                # 根据结果获得设备故障时间
                cur['TIME'] = cur['TIME'].apply(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d %H:%M:%S'))
                cur = cur[cur['result'] == 1]
                cur_result = pd.DataFrame()
                cur_result['ERR_TIME'] = cur['TIME'].values.tolist()
                cur_result['result'] = cur['result'].values.tolist()
                cur_result['DEV_TYPE'] = row['DEV_TYPE']
                cur_result['PRODUCER_CODE'] = row['PRODUCER_CODE']
                cur_result['PAR_ROOM'] = row['PAR_ROOM']
                cur_result['CITY'] = city
                feature = feature.append(cur_result)

        if not os.path.exists(root + '/info/predict_dev_err.csv'):
            feature.to_csv(root + '/info/predict_dev_err.csv', encoding='utf-8', index=False, header=True)
        else:
            feature.to_csv(root + '/info/predict_dev_err.csv', encoding='utf-8', index=False, header=False, mode='a')


def get_true_alert(start, end):
    data = pd.read_csv(root + '/info/room_err_time.csv', encoding='utf-8')
    data = data[(data['ALARM_EMS_TIME_DAY'] >= start) & (data['ALARM_EMS_TIME_DAY'] <= end)]
    all_dev = pd.read_csv(root + '/info/dev_info.csv', encoding='utf-8')
    data = pd.merge(data, all_dev, how='left')
    print(data.shape[0])
    print(data.values)


# 数据源时间: 2016/01/03 至 2018/04/20
# 使用2016/01/03 至 2017/10/01 训练模型  余下数据做预测验证
if __name__ == '__main__':
    common = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    root = os.path.abspath(os.path.dirname(__file__))
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
    start = '2017-10-01'
    end = '2017-10-07'

    gen_demand_by_par_room()
    #predict_parts_change(start, end)
    #get_true_alert(start, end)
