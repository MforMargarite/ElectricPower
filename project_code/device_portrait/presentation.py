import pandas as pd
import os
import datetime
import time
import json
import numpy as np


def room_for_presentation():
    data = pd.read_csv(root + '/info/room_err_time.csv', encoding='utf-8')
    # data = data[data['ALARM_CAUSE'].isin()]
    rank_by_par_room = data.groupby('PAR_ROOM', as_index=False).size()
    rank_by_par_room = pd.DataFrame(rank_by_par_room, columns=['count'])
    rank_by_par_room.reset_index(inplace=True)
    rank_by_par_room.sort_values(by='count', ascending=False, inplace=True)
    potentials = rank_by_par_room['PAR_ROOM'].values.tolist()

    attrs = [i for i in os.listdir(parent + '/data/') if i != 'raw_data']

    all_exists = None
    for att in attrs:
        rs = [i for i in os.listdir(parent + '/data/' + att + '/') if '.' not in i]
        if all_exists is None:
            all_exists = set(rs)
        else:
            all_exists = all_exists & set(rs)

    result = []
    for i in potentials:
        if i in all_exists:
            result.append(i)
        if len(result) >= 3:
            break
    return result


def get_time_before(x, hour):
    hour *= -1
    before = x + datetime.timedelta(hours=hour)
    t = before.timetuple()
    return time.strftime('%Y-%m-%d %H:00:00', t)


def triphase_transform(c):
    if '_' in c:
        filtered = c.split("_")
        if '_A' in c or '_B' in c or '_C' in c:
            if len(filtered) == 4:
                return filtered[1] + "_" + filtered[3]
            else:
                return filtered[1] + "_" + filtered[2]
        else:
            return filtered[1]
    return c


def gen_label(threshold, data, col):
    result = []
    for index, row in data.iterrows():
        each_row = ''
        for c in col:
            if 'DIFF_RATE' not in c and 'VALUE' not in c:
                _range = threshold.get(triphase_transform(c))
                if not np.isnan(row[c]):
                    if row[c] < _range[0]:
                        each_row += str(24 - index)+'_min_' + c + '|'
                    elif row[c] > _range[1]:
                        each_row += str(24 - index)+'_max_' + c + '|'
        if each_row != '':
            result.append(each_row[:-1])
        else:
            result.append(np.NaN)
    return result


def valid_data(series):
    return series.dropna().shape[0] == 24


def get_info(rooms):
    if not os.path.exists(root + '/presentation/'):
        os.mkdir(root + '/presentation/')
    if not os.path.exists(root + '/presentation/js/'):
        os.mkdir(root + '/presentation/js/')
    err_time = pd.read_csv(root + '/info/room_err_time.csv', encoding='utf-8', usecols=['PAR_ROOM', 'ALARM_EMS_TIME'])
    err_time['ALARM_EMS_TIME'] = err_time['ALARM_EMS_TIME'].apply(lambda x: datetime.datetime.strftime(datetime.datetime.fromtimestamp(x / 1000).replace(minute=0, second=0), '%Y-%m-%d %H:%M:%S'))
    normal_t, normal_c, normal_v = json.load(open(parent + '/data/temperature/normal_feature.txt', 'r')), json.load(open(parent + '/data/current/normal_feature.txt', 'r')),json.load(open(parent + '/data/voltage/normal_feature.txt', 'r'))

    to_js = open(root + '/presentation/js/device_portrait.js', 'w')
    r_num = 0
    for r in rooms:
        if not os.path.exists(root + '/presentation/' + r + '/'):
            os.mkdir(root + '/presentation/' + r + '/')
        result = pd.DataFrame()
        # 获取机房故障时间和属性序列
        room_err_time = err_time[err_time['PAR_ROOM'] == r]
        room_err_time.reset_index(inplace=True)
        room_t, room_c, room_v = pd.read_csv(parent + '/data/temperature/' + r + '/data_clean.csv', encoding='utf-8'), pd.read_csv(parent + '/data/current/' + r + '/data_clean.csv', encoding='utf-8'), pd.read_csv(parent + '/data/voltage/' + r + '/data_clean.csv', encoding='utf-8')
        room_t, room_c, room_v = room_t.drop(columns=['result'], axis=1), room_c.drop(columns=['result'], axis=1), room_v.drop(columns=['result'], axis=1)
        room_t.set_index('TIME', inplace=True)
        room_c.set_index('TIME', inplace=True)
        room_v.set_index('TIME', inplace=True)

        for t in room_err_time['ALARM_EMS_TIME'].values.tolist():
            # 获取前24h时序
            before_24 = get_time_before(pd.datetime.strptime(t, '%Y-%m-%d %H:%M:%S'), 24)
            if before_24 > '2016-11-01 00:00:00':
                before_1 = get_time_before(pd.datetime.strptime(t, '%Y-%m-%d %H:%M:%S'), 1)
                # 温度、电压、电流时序
                t_series, c_series, v_series = pd.DataFrame(room_t[before_24:before_1].values, columns=list(map(lambda x: 'T_' + x, room_t.columns.tolist()))), pd.DataFrame(room_c[before_24:before_1].values, columns=list(map(lambda x: 'C_' + x, room_c.columns.tolist()))), pd.DataFrame(room_v[before_24:before_1].values, columns=list(map(lambda x: 'V_' + x, room_v.columns.tolist())))
                if valid_data(t_series) and valid_data(c_series) and valid_data(v_series):
                    t_label, c_label, v_label = gen_label(normal_t, t_series, t_series.columns.tolist()), gen_label(normal_c, c_series, c_series.columns.tolist()), gen_label(normal_v, v_series, v_series.columns.tolist())
                    cur = pd.concat([t_series, c_series, v_series], axis=1)
                    cur.reset_index(inplace=True)
                    cur['t_label'] = t_label
                    cur['c_label'] = c_label
                    cur['v_label'] = v_label
                    if len(set(t_label)) > 1 or len(set(c_label)) > 1 or len(set(v_label)) > 1:
                        cur['before_err'] = list(range(24, 0, -1)) * int(cur.shape[0] / 24) + list(range(24, 24 - int(cur.shape[0] % 24), -1))
                        cur.drop(columns=['index'], axis=1, inplace=True)
                        result = result.append(cur)
        result.to_json(root + '/presentation/' + r + '/device_portrait.json', orient='table')
        to_js.write('var info_' + str(r_num) + '= '+json.dumps(json.load(open(root + '/presentation/' + r + '/device_portrait.json', 'r'))["data"]) + ';')
        to_js.flush()
        os.remove(root + '/presentation/' + r + '/device_portrait.json')
        r_num += 1
    to_js.write('var r_num = ' + str(r_num) + ';')
    to_js.flush()
    to_js.close()


if __name__ == '__main__':
    parent = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    root = os.path.abspath(os.path.dirname(__file__))
    rooms = room_for_presentation()
    get_info(rooms)
