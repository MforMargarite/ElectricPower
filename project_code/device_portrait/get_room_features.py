#  数据获取
#  不同机房下的温度/电压/电流变化情况及机房-设备对应关系等基本信息


import pandas as pd
import pymysql
import os
import datetime
import time
from sklearn import preprocessing


def gen_room_attr(path, c, att, col):
    connection = pymysql.connect(**c)
    try:
        mapping = pd.read_csv(path + '/data/' + att + '.csv', encoding='utf-8')
    except:
        mapping = pd.read_csv(path + '/data/' + att + '.csv', encoding='gbk')
    last_obj = '0'
    output = pd.DataFrame()
    mapping = mapping.sort_values(by=[col], ascending=True)
    mapping['RES_OBJ_NAME'] = mapping['RES_OBJ_NAME'].apply(lambda x: x.replace('\\', '%'))
    for index, row in mapping.iterrows():
        if os.path.exists(path + '/data/' + att + '/' + row[col] + '.csv'):
            continue
        try:
            t_sql = "select time,value,RES_OBJ_NAME from rt_cv_1003,rt_cv_general_last where CURVE_NO = TABLE_NAME and RES_OBJ_NAME like '%" + row['RES_OBJ_NAME'] +"%'"
            df = pd.read_sql(t_sql, connection)
            df.drop_duplicates(inplace=True)
            df.columns = ['TIME', 'VALUE', 'RES_OBJ_NAME']
            if last_obj != '0' and row[col] != last_obj:
                output.to_csv(path + '/data/' + att + '/' + last_obj + '.csv', encoding='utf-8', header=True, index=False)
                output = pd.DataFrame()
            output = output.append(df)
            last_obj = row[col]
        except:
            print("err")
    connection.close()


# 获取设备和房间的对应关系   2016/9开始
def get_room_device_mapping(path, c):
    try:
        room_info = pd.read_csv(path + '/trend/temperature.csv', encoding='utf-8')
    except UnicodeEncodeError:
        room_info = pd.read_csv(path + '/trend/temperature.csv', encoding='gbk')
    c['db'] = 'jiangxi'
    connection = pymysql.connect(**c)
    try:
        t_sql = "select OBJ_ID,PAR_ROOM from t_ne where PAR_ROOM in " + str(tuple(room_info['PAR_ROOM'].values.tolist()))
        df = pd.read_sql(t_sql, connection)
        df.drop_duplicates(inplace=True)
        df.columns = ['NE_OBJ_ID', 'PAR_ROOM']
        df.to_csv(root+'/trend/room_device_mapping.csv', encoding='utf-8', header=True, index=False)
    finally:
        connection.close()


def get_time_before(x, hour):
    hour *= -1
    x /= 1000
    before = datetime.datetime.fromtimestamp(x) + datetime.timedelta(hours=hour)
    t = before.timetuple()
    return int(time.mktime(t))


# 获取机房设备发生故障的时间
def get_err_time(path, hour):
    try:
        device_err = pd.read_csv(path+'/pinshan/jiangxi_electric' + str(hour) + 'h_data_clean.csv', encoding='utf-8')
    except UnicodeDecodeError:
        device_err = pd.read_csv(path + '/pinshan/jiangxi_electric' + str(hour) + 'h_data_clean.csv', encoding='gbk')
    room_device = pd.read_csv(path+'/trend/room_device_mapping.csv', encoding='utf-8')
    device_err = pd.merge(device_err, room_device, on=["NE_OBJ_ID"], how='left')
    device_err = device_err.loc[:, ['NE_OBJ_ID', 'ALARM_EMS_TIME_DAY', 'ALARM_EMS_TIME', 'ALARM_EMS_RESUME_TIME', 'ALARM_CAUSE', 'PAR_ROOM']]

    device_err = device_err[device_err['ALARM_EMS_TIME_DAY'] < '2018-04-19']
    need_change_dict = pd.read_csv(root + '/info/err_type.csv', encoding='utf-8', index_col='err_type').to_dict()['need_change']
    device_err['NEED_CHANGE'] = device_err['ALARM_CAUSE'].apply(lambda x: need_change_dict[x] + 1)
    device_err['ALARM_EMS_TIME_DAY'] = device_err['ALARM_EMS_TIME_DAY'].apply(lambda x: pd.datetime.strftime(pd.datetime.strptime(x, '%Y/%m/%d'),  '%Y-%m-%d'))
    device_err.sort_values(by=['PAR_ROOM', 'ALARM_EMS_TIME_DAY'], ascending=True, inplace=True)
    device_err.to_csv(path+'/info/room_err_time.csv', encoding='utf-8', header=True, index=False)


# 获取各机房每日各型号的需求量
# 生成room_day_demand.csv
def gen_demand_by_par_room():
    if not os.path.exists(root + '/info/room_day_demand.csv'):
        err_time = pd.read_csv(root + '/info/room_err_time.csv', encoding='utf-8')
        err_time = err_time[err_time['NEED_CHANGE'] == 2]
        dev_info = pd.read_csv(root + '/info/dev_info.csv', encoding='utf-8')
        err_time = pd.merge(err_time, dev_info, on=['NE_OBJ_ID', 'PAR_ROOM'], how='left')
        # 按设备ID、故障日期分组
        err_time = err_time.groupby(['DEV_TYPE', 'PRODUCER_CODE', 'PAR_ROOM', 'NE_OBJ_ID', 'ALARM_EMS_TIME_DAY'],as_index=False).size()
        err_time = pd.DataFrame(err_time, columns=['count'])
        err_time.reset_index(inplace=True)
        # 同一设备同一天的故障视作一个
        err_time.drop(columns='count', axis=1, inplace=True)
        # 按机房、设备型号、故障日期分组求和
        err_time = err_time.groupby(['DEV_TYPE', 'PRODUCER_CODE', 'PAR_ROOM', 'ALARM_EMS_TIME_DAY'], as_index=False).size()
        err_time = pd.DataFrame(err_time, columns=['demand'])
        err_time.reset_index(inplace=True)
        err_time['MONTH'] = err_time['ALARM_EMS_TIME_DAY'].apply(lambda x: x[:-3])
        # 地理位置信息
        err_time = gen_demand_by_city(err_time)
        err_time.to_csv(root + '/info/room_day_demand.csv', encoding='utf-8', header=True, index=False)


# 根据机房地理位置映射 汇总各市(区)零件需求
def gen_demand_by_city(df):
    geo_info = pd.read_csv(root + '/info/t_spc_room.csv', encoding='gbk', usecols=['OBJ_ID', 'LOCATION'])
    merged = pd.merge(df, geo_info, how='left', left_on='PAR_ROOM', right_on='OBJ_ID')
    merged.drop(columns=['OBJ_ID'], axis=1, inplace=True)
    return merged


if __name__ == '__main__':
    root = os.path.abspath(os.path.dirname(__file__))
    config = {
        'host': "172.16.135.6",
        'port': 3306,
        'user': 'root',
        'password': '10086',
        'db': 'jiangxi16.07-17.06',
        'charset': 'utf8',
        'cursorclass': pymysql.cursors.DictCursor,
    }
    #get_room_device_mapping(root, config)
    get_err_time(root, float(24))
    gen_demand_by_par_room()
    #gen_room_attr(root, config, 'voltage', 'OBJ_ID')
    #gen_room_attr(root, config, 'current', 'OBJ_ID')
    #gen_room_attr(root, config, 'temperature', 'PAR_ROOM')
