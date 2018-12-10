# 对由数据库获得的机房属性数据进行清洗 整理
# 用分布拟合了故障时间和正常时间下的机房属性 为故障时间下的机房属性打标签
# 进行关联分析 找到出现故障的置信度较高的模式
import orangecontrib.associate.fpgrowth as oaf       # pip install orange3-associate
import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime
import time
import numpy as np
import json


def remain_float(x):
    return [i for i in x if str(i).split(".")[1] != '0']


def pre_process(path, att):
    room_dev_data = pd.read_csv(common + 'room_device_mapping.csv', encoding='utf-8')
    err_time = pd.read_csv(common + 'room_err_time.csv', encoding='utf-8')
    rooms = room_dev_data['PAR_ROOM'].unique()
    for r in rooms:
        room_err_time = list(map(lambda x: datetime.datetime.strftime(datetime.datetime.fromtimestamp(x / 1000).replace(minute=0, second=0), '%Y-%m-%d %H:%M:%S'),err_time[err_time['PAR_ROOM'] == r]['ALARM_EMS_TIME'].values.tolist()))
        if os.path.exists(path + '/data/raw_data/' + att + '/' + r + '.csv'):
            data = pd.read_csv(path + '/data/raw_data/' + att + '/' + r + '.csv', encoding='utf-8')
            # 剔除异常记录
            time_series = data[data['VALUE'].isin(remain_float(data['VALUE'].values.tolist()))]
            time_series['TIME'] = time_series['TIME'].apply(lambda x: datetime.datetime.strftime(datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').replace(minute=0, second=0), '%Y-%m-%d %H:%M:%S'))
            time_range = time_series['TIME'].unique().tolist()
            time_range.sort()
            start = time_range[0]
            end = time_range[-1]
            # 平滑处理 取1h均值和方差
            df_mean = time_series.groupby('TIME').mean().reset_index().loc[:, ['TIME', 'VALUE']]  # 均值
            df_mean['DIFF'] = gen_diff(df_mean['VALUE'])                                          # 变化量
            df_mean['DIFF_RATE'] = gen_diff_rate(df_mean['VALUE'])                                 # 变化率
            df_var = time_series.groupby('TIME').std().reset_index().loc[:, ['TIME', 'VALUE']]    # 方差
            df_var.rename(columns={'VALUE': 'VAR'}, inplace=True)

            result = pd.DataFrame()
            result['TIME'] = pd.date_range(start, end, freq='h').astype(str)
            result = pd.merge(result, df_mean, how='left', on='TIME')
            result = pd.merge(result, df_var, how='left', on='TIME')
            result['result'] = result['TIME'].apply(lambda x: 1 if x in room_err_time else 0)
            # 保存至文件
            if not os.path.exists(path + 'data/' + att + '/'):
                os.mkdir(path + 'data/' + att + '/')
            if not os.path.exists(path + 'data/' + att + '/' + r):
                os.mkdir(path + 'data/' + att + '/' + r)
            if len(result['result'].unique().tolist()) == 1:
                result.to_csv(path + '/data/' + att + '/' + r + '/normal_data_clean.csv', header=True, index=False, encoding='utf-8')
            else:
                result.to_csv(path + '/data/' + att + '/' + r + '/data_clean.csv', header=True, index=False, encoding='utf-8')


# 电流与电压 根据所属机房整合
def combine_by_room(path, att):
    room_dev_data = pd.read_csv(path + '/room_device_mapping.csv', encoding='utf-8')
    err_time = pd.read_csv(path + '/room_err_time.csv', encoding='utf-8')
    rooms = room_dev_data['PAR_ROOM'].unique()
    for r in rooms:
        devices = room_dev_data[room_dev_data['PAR_ROOM'] == r]['NE_OBJ_ID'].unique().tolist()
        devices.sort()
        room_err_time = list(map(lambda x: datetime.datetime.strftime(datetime.datetime.fromtimestamp(x / 1000).replace(minute=0, second=0), '%Y-%m-%d %H:%M:%S'), err_time[err_time['PAR_ROOM'] == r]['ALARM_EMS_TIME'].values.tolist()))

        combined = None
        dev_num = 0
        for dev in devices:
            print(dev)
            if os.path.exists(path + '/data/raw_data/' + att + '/' + dev + '.csv'):
                df = pd.read_csv(path + '/data/raw_data/' + att + '/' + dev + '.csv', encoding='utf-8', date_parser=lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').replace(second=0,minute=0), parse_dates=['TIME'])
                # 去除符号影响 符号表示方向
                df['VALUE'] = df['VALUE'].apply(lambda x: np.fabs(x))
                df['RES_OBJ_NAME'] = df['RES_OBJ_NAME'].astype(str)
                df['TIME'] = df['TIME'].astype(str)
                # 生成时间范围
                time_range = df['TIME'].unique().tolist()
                time_range.sort()
                start = time_range[0]
                end = time_range[-1]

                result = pd.DataFrame()
                result['TIME'] = pd.date_range(start, end, freq='h').astype(str)
                # 获取不同的机房属性
                category = df['RES_OBJ_NAME'].unique().tolist()
                category.sort()
                print(category)
                for c in category:
                    d_data = df[df['RES_OBJ_NAME'] == c].reset_index().loc[:, ['TIME', 'VALUE']]

                    mean = d_data.groupby('TIME').mean().reset_index().loc[:, ['TIME', 'VALUE']]            # 均值
                    mean['DIFF'] = gen_diff(mean['VALUE'])                                                  # 变化量
                    mean['DIFF_RATE'] = gen_diff_rate(mean['VALUE'])                                        # 变化率
                    var = d_data.groupby('TIME').std().reset_index().loc[:, ['TIME', 'VALUE']]              # 方差

                    # 获取三相电压的信息
                    pos = c.find('相')
                    if pos != -1:
                        tri_phase = c[pos-1:pos]
                        mean.rename(columns={'VALUE': 'VALUE_' + str(dev_num) + '_' + tri_phase, 'DIFF': 'DIFF_' + str(dev_num) + '_' + tri_phase, 'DIFF_RATE': 'DIFF_RATE_' + str(dev_num) + '_' + tri_phase}, inplace=True)
                        var.rename(columns={'VALUE': 'VAR_' + str(dev_num) + '_' + tri_phase}, inplace=True)
                    else:
                        mean.rename(columns={'VALUE': 'VALUE_' + str(dev_num), 'DIFF': 'DIFF_' + str(dev_num), 'DIFF_RATE': 'DIFF_RATE_' + str(dev_num)}, inplace=True)
                        var.rename(columns={'VALUE': 'VAR_' + str(dev_num)}, inplace=True)

                    result = pd.merge(result, mean, how='left', on='TIME')
                    result = pd.merge(result, var, how='left', on='TIME')
                if combined is None:
                    combined = result
                else:
                    new_combined = pd.merge(combined, result, how='outer', on=['TIME'])
                    combined = new_combined
                dev_num += 1
        if combined is not None:
            combined['result'] = combined['TIME'].apply(lambda x: 1 if x in room_err_time else 0)
            if not os.path.exists(path + 'data/' + att + '/'):
                os.mkdir(path + 'data/' + att + '/')
            if not os.path.exists(path + 'data/' + att + '/' + r):
                os.mkdir(path + 'data/' + att + '/' + r)
            if len(combined['result'].unique().tolist()) == 1:
                combined.to_csv(path + '/data/' + att + '/' + r + '/normal_data_clean.csv', header=True, index=False, encoding='utf-8')
            else:
                combined.to_csv(path + '/data/' + att + '/' + r + '/data_clean.csv', header=True, index=False, encoding='utf-8')


def get_time_before(x, hour):
    hour *= -1
    before = x + datetime.timedelta(hours=hour)
    t = before.timetuple()
    return time.strftime('%Y-%m-%d %H:00:00', t)


def inflexion_rate(x, y):
    if pd.isnull(x) or pd.isnull(y):
        return 'a'
    elif x == 0:
        return y - x
    return (y - x) / x


def inflexion(x, y):
    if pd.isnull(x) or pd.isnull(y):
        return 'a'
    return y - x


def gen_diff_rate(_value):
    _tuple = zip(_value[:-1], _value[1:])
    # 计算变化趋势(xi - xi-1)/xi-1
    diff = [0]
    for tu in _tuple:
        diff.append(inflexion_rate(*tu))
    return diff


def gen_diff(_value):
    _tuple = zip(_value[:-1], _value[1:])
    # 计算变化趋势(xi - xi-1)/xi-1
    diff = [0]
    for tu in _tuple:
        diff.append(inflexion(*tu))
    return diff


def rule_process(rules, err_type):
    clean_rules = []
    for i in rules:
        condition = ''
        rule = ''
        cur_rule_list = []
        for j in i[0]:
            condition = condition + ' & ' + j
        condition = condition[3:] + ' ==> '

        for j in i[1]:
            rule = rule + ' & ' + j
        rule = rule[3:]

        for t in err_type:
            if t == rule:
                cur_rule_list.append(condition + rule)
                cur_rule_list.append(i[3])
                clean_rules.append(cur_rule_list)
                break
    return clean_rules


def to_label(col, data, time):
    result = []
    for index, row in data.iterrows():
        each_row = ''
        for c in col:
            if 'DIFF_RATE' not in c:
                if row[c] == -1:
                    each_row += str(time[index])+'_min_' + c + '|'
                elif row[c] == 1:
                    each_row += str(time[index])+'_max_' + c + '|'
                # else:
                #     each_row += str(time[index])+'_normal_' + c + '|'
        if each_row != '':
            result.append(each_row[:-1])
        else:
            result.append(np.NaN)
    return result


def asso_analysis(path, file_path):
    if not os.path.exists(root + '/asso_analysis/'):
        os.mkdir(root + '/asso_analysis/')

    if not os.path.exists(root + '/asso_analysis/err_label_clean.csv'):
        data = pd.read_csv(path + file_path, encoding='utf-8')
        room_list = data['PAR_ROOM'].unique().tolist()
        room_type_map = {}
        # 生成机房类型 - 机房ID映射字典
        for r in room_list:
            r_info = data[data['PAR_ROOM'] == r].dropna(axis=1)
            if str(r_info.columns.tolist()) not in room_type_map.keys():
                room_type_map[str(r_info.columns.tolist())] = [r]
            else:
                room_type_map[str(r_info.columns.tolist())].append(r)

        # 生成标记数据
        label_df = pd.DataFrame()
        for k in room_type_map.keys():
            same_type_room = room_type_map.get(k)
            df = pd.DataFrame()
            for r in same_type_room:
                df = df.append(data[data['PAR_ROOM'] == r], ignore_index=True)
            df.dropna(axis=1, inplace=True)

            col = [c for c in df.columns.tolist() if c not in ['TIME','PAR_ROOM','ALARM_CAUSE']]
            cur_label = pd.DataFrame()
            cur_label['before_err'] = list(range(24, 0, -1)) * int(df.shape[0] / 24) + list(range(24, 24 - int(df.shape[0] % 24), -1))
            cur_label['err_feature'] = to_label(col, df, cur_label['before_err'].values.tolist())
            cur_label['ALARM_CAUSE'] = df['ALARM_CAUSE']
            label_df = label_df.append(cur_label)
        label_df.dropna(inplace=True)
        label_df.to_csv(root + '/asso_analysis/err_label_clean.csv', index=False, encoding='utf-8')
    cur_cate = pd.read_csv(root + '/asso_analysis/err_label_clean.csv', encoding='utf-8', low_memory=False)

    cur_cate.dropna(inplace=True)
    cate_dict = {'R_LOS': 161, 'NE_NOT_LOGIN': 161, 'High Temperature': 161, 'NE_COMMU_BREAK': 161, 'lossOfSignal': 161, 'R_LOF': 161, 'IN_PWR_HIGH': 161, 'POWERALM': 161, 'HARD_BAD': 161,
                 'NE_Backup_Failed': 161, 'Comms fail alarm': 161, 'FCS_ERR': 161, 'LSR_NO_FITED': 161, 'PKG_FAIL': 161, 'IN_PWR_FAIL': 161, 'BUS_ERR': 161, 'PLUGGABLE_TRANSCEIVER_DISMOUNT': 161,
                 'R_OOF': 161, 'PWR_MAJ_ALM': 161, 'Client Service Mismatch': 161, 'UNKNOWN_CARD': 161, 'OS-Optical_Power_High': 161, 'GNE_CONNECT_FAIL': 161,
                 'Replaceable Unit Problem': 162, 'Loss Of Signal': 162, 'LOS': 162, 'LOF': 162, 'IN_PWR_ABN': 162, 'OUT_PWR_ABN': 162, 'Underlying Resource Unavailable': 162, 'Loss Of Frame': 162,
                 'ME loss of communication': 162, 'COMMUN_FAIL': 162, 'TEMP_OVER': 162, 'BD_STATUS': 162, 'SUBCARD_ABN': 162, 'POWER_FAIL': 162, 'Duplicate Shelf Detected': 162,
                 'NE_DATA_INCONSISTENCY': 162, 'SYSBUS_FAIL': 162, 'SHELF_ABSENCE': 162, 'ABSENCE_WARNING': 162, 'POWER_ABNORMAL': 162, 'Bipolar Violations': 162, 'Transmitter Failure': 162, 'CHIP_FAIL': 162,
                 'BUS_ERROR': 162, 'LAPS_FAIL': 162, 'Degraded Signal': 163, 'Signal Degrade': 163, 'Internal Communication Problem': 163, 'RDI': 163,
                 'cntrlBusFail': 163, 'BD_NOT_INSTALLED': 163, 'FAN_FAIL': 163, 'SYN_BAD': 163, 'Circuit Pack Mismatch': 163, 'Fan Failed': 163, 'Replaceable Unit Missing': 163,
                 'Fuse Failure': 163, 'Battery Failure': 163, 'Temperature Out Of Range': 163, 'Power Failure - B': 163, 'Database Save and Restore Failed': 163, 'Cooling Fan Failure': 163,
                 'MIB backup misaligned': 164, 'Inside Failure': 164, 'Sfwr Environment Problem': 164, 'HouseKeeping': 164}

    err_type = ['161', '162', '163', '164']
    # err_type = cur_cate['ALARM_CAUSE'].unique().tolist()
    cur_cate['ALARM_CAUSE'] = cur_cate['ALARM_CAUSE'].apply(lambda x: str(cate_dict[x]) if x in cate_dict.keys() else "-1")
    cur_cate['err_feature'] = cur_cate['err_feature'].apply(lambda x: x.split("|"))

    err_feature = []
    last_before_err = 24
    items_dict = {'161': [], '162': [], '163': [], '164': []}
    for index, row in cur_cate.iterrows():
        err_feature.append(row['err_feature'])
        if last_before_err < row['before_err'] or index == cur_cate.shape[0]-1:
            cause = cur_cate.loc[index - 1, 'ALARM_CAUSE']
            items_dict[cause] += err_feature
            err_feature.clear()
        last_before_err = row['before_err']

    d_itemsets = {}
    for c in err_type:
        # 频繁项集
        each_itemsets = dict(oaf.frequent_itemsets(items_dict[c], 0.0125))
        total = 0
        # 关联规则
        for k in each_itemsets.keys():
            s = set(k)
            s.add(c)
            d_itemsets[frozenset(s)] = each_itemsets[k]
            if k not in d_itemsets:
                d_itemsets[k] = each_itemsets[k]
            else:
                d_itemsets[k] += each_itemsets[k]
            total += each_itemsets[k]
        d_itemsets[frozenset([c])] = total
    rules = list(oaf.association_rules(d_itemsets, 0.7))
    cur_result = pd.DataFrame(rule_process(rules, err_type), columns=('规则', '置信度'))
    cur_result.to_csv(root + '/asso_analysis/associate_analysis.csv', encoding='utf-8', header=True, index=False)


def gen_err_norm_room(path, att):
    _err = []
    _normal = []
    for room in [f for f in os.listdir(path + '/data/' + att + '/') if '.' not in f]:
        if 'normal_data_clean.csv' in os.listdir(path + '/data/' + att + '/' + room + '/'):
            _normal.append(room)
        else:
            _err.append(room)
    return _err, _normal


def get_before_time(t, delta):
    delta *= -1
    d = datetime.datetime.fromtimestamp(t).replace(minute=0, second=0)
    return d + datetime.timedelta(hours=delta), pd.date_range(d + datetime.timedelta(hours=delta), d, freq='h', closed='left').astype(str)


def triphase_transform(c):
    if '_' in c:
        filtered = c.split("_")
        if '_A' in c or '_B' in c or '_C' in c:
            return filtered[0] + "_" + filtered[2]
        elif '0' in c or '1' in c or '2' in c:
            return filtered[0]
    return c


# 获取正常机房的各属性变化范围
def get_normal_range(rooms, att, kind):
    result_map = {}
    for r in rooms:
        data = pd.read_csv(common + '/data/' + att + '/' + r + '/' + kind + '.csv', encoding='utf-8')
        for c in data.columns.tolist():
            if c != 'result' and c != 'TIME' and 'VALUE' not in c and 'DIFF_RATE' not in c:
                map_c = triphase_transform(c)
                if map_c in result_map.keys():
                    val = result_map.get(map_c)
                    val += list(filter(lambda x: not pd.isnull(x), data[c].values.tolist()))
                else:
                    val = list(filter(lambda x: not pd.isnull(x), data[c].values.tolist()))
                result_map[map_c] = val
    for k in result_map.keys():
        result_map[k] = gen_attr_threshold(att, k, result_map.get(k))
    # 属性正常取值范围 保存至文件
    json_obj = json.dumps(result_map)
    to_file = open(common + '/data/' + att + '/normal_feature.txt', 'w')
    to_file.write(json_obj)
    to_file.close()
    return result_map


def gen_attr_threshold(att, k, val):
    # 绘制累积曲线
    fig = plt.figure()
    bins, patches = np.histogram(val, bins=100, density=True)
    plt.hist(val, bins=100, density=True)
    plt.xlim([-3, 3])
    fig.savefig(common + '/data/' + att + '/' + k.lower() + '_prob.png', transparent=True, dpi=80)
    fig.clear()

    # 求95%置信区间
    start = 0
    end = len(bins)
    while np.sum(np.diff(patches[start:end]) * bins[start:end-1]) > 0.95:
        if 'var' in k.lower():
            end -= 1
        else:
            if bins[start] < bins[end-1]:
                start += 1
            else:
                end -= 1
    return [patches[start], patches[end]]


def gen_label(val, col, range):
    r = []
    for index, row in val.iterrows():
        v = row[col]
        if pd.isnull(v):
            r.append(-2)
        elif v >= range[1]:
            r.append(1)
        elif v < range[0]:
            r.append(-1)
        else:
            r.append(0)
    return r


# 获取故障机房对应故障时间前n小时 某一属性的变化率和方差
def get_attr_series(attr, n, normal_map):
    err_time = pd.read_csv(common + '/room_err_time.csv', encoding='utf-8')
    res = pd.DataFrame()

    for index, row in err_time.iterrows():
        if os.path.exists(common + '/data/' + attr + '/' + row['PAR_ROOM'] + '/data_clean.csv'):
            room_data = pd.read_csv(common + '/data/' + attr + '/' + row['PAR_ROOM'] + '/data_clean.csv')
            t = row['ALARM_EMS_TIME']
            begin, watch_time = get_before_time(t/1000, n)
            data = pd.DataFrame()
            data['TIME'] = watch_time
            if timestamp_parse(t) in room_data['TIME'].values.tolist() and datetime.datetime.strftime(begin, '%Y-%m-%d %H:%M:%S') in room_data['TIME'].values.tolist():
                err_room_data = room_data[room_data['TIME'].isin(watch_time.astype(str))]
                attr_alias = attr[0:1].upper()
                cols = [c for c in room_data.columns.tolist() if triphase_transform(c) in normal_map.keys()]
                for c in cols:
                    data[attr_alias + '_' + c] = gen_label(err_room_data, c, range=normal_map.get(triphase_transform(c)))
                data['PAR_ROOM'] = row['PAR_ROOM']
                #data['ALARM_CAUSE'] = cate_dict[row['ALARM_CAUSE']] if row['ALARM_CAUSE'] in cate_dict.keys() else np.NaN
                data['ALARM_CAUSE'] = row['ALARM_CAUSE']
                res = res.append(data)
    return res


def gen_room_err_feature():
    if not os.path.exists(root + '/asso_analysis/'):
        os.mkdir(root + '/asso_analysis/')
    if not os.path.exists(root + '/asso_analysis/err_feature.csv'):
        # 遍历机房属性 分析
        result = None
        for attr in r_attr + d_attr:
            # 获取故障机房和正常机房
            err_room, normal_room = gen_err_norm_room(common, attr)
            # 获取正常机房的各属性变化范围
            attr_map = get_normal_range(normal_room, attr, 'normal_data_clean')
            # 获取故障机房中 故障时间前hour小时的属性时序  标签化
            if result is None:
                result = get_attr_series(attr, hour, attr_map)
            else:
                result = pd.merge(result, get_attr_series(attr, hour, attr_map), how='inner',
                                  on=['TIME', 'PAR_ROOM', 'ALARM_CAUSE'])
        result.to_csv(root + '/asso_analysis/err_feature.csv', encoding='utf-8', header=True, index=False)


if __name__ == '__main__':
    # 数据源时间: 16.09至18.04
    root = os.path.abspath(os.path.dirname(__file__)) + '/'
    common = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/'
    d_attr = ['current', 'voltage']         # 以设备为单位获取的属性信息 如电压、电流
    r_attr = ['temperature']                # 以机房为单位获取的属性信息 如温度
    aim_col = 'PAR_ROOM'
    hour = 24
    # 与日期处理有关的参数
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
    timestamp_parse = lambda t: datetime.datetime.strftime(datetime.datetime.fromtimestamp(t/1000).replace(minute=0, second=0), '%Y-%m-%d %H:%M:%S')
    # 数据预处理
    # for a in r_attr:
    #     pre_process(root, a)
    # for a in d_attr:
    #     combine_by_room(common, a)
    # 生成机房故障时序对应属性的标记
    gen_room_err_feature()
    # 关联分析
    asso_analysis(root, '/asso_analysis/err_feature.csv')




