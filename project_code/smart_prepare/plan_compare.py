import pandas as pd
import os
import numpy as np
import copy
import pymysql
# 对比原有和新提出的智能备件方案
# 以江西省为例  以2017.10至2017.12故障记录为数据源，计算成本、运输时间
# 原有备件方案：保证每种型号的备件个数维持在总个数的10%(小于10个配1个)    备件中心：南昌

# 备件方案成本和耗时计算方法：
# 成本： 基本备件量的管理成本 + 更换的设备价格 + 运输费用
# 耗时:  运输时间 + 可能的提货等待时间


# 获取原有方案的初始化(基本)备件需求量 存至字典中
def gen_old_plan_base_demand():
    if not os.path.exists(root + '/info/old_prepare_plan.csv'):
        dev_info = pd.read_csv(root + '/info/dev_info.csv', encoding='utf-8')
        dev_info['TYPE'] = dev_info.apply(lambda row: str(int(row['PRODUCER_CODE'])) + "_" + str(int(row['DEV_TYPE'])), axis=1)
        dev_count = dev_info.groupby('TYPE', as_index=False).size()
        dev_count = pd.DataFrame(dev_count, columns=['count'])
        dev_count.reset_index(inplace=True)
        dev_count['count'] = dev_count['count'].apply(lambda x: round(0.1 * x) if x >= 10 else 1)
        dev_count.to_csv(root + '/info/old_prepare_plan.csv', encoding='utf-8', header=True, index=False)
    else:
        dev_count = pd.read_csv(root + '/info/old_prepare_plan.csv', encoding='utf-8')
    dev_count.set_index('TYPE', inplace=True)
    return dev_count.to_dict()['count']


# 返回值： dict  格式: {型号:{备件中心：{月份：需求量}}}
# 写入数据库 t_smart_prepare表
def gen_new_plan_demand(start, end, threshold):
    demand_dict = {}
    data = pd.read_csv(root + '/info/parts_prepare.csv', encoding='utf-8', usecols=['city', 'dev_type', 'predict_by_month', 'start_month', 'end_month', 'well_learnt'])
    center_num = len(data['city'].unique().tolist()) / 2
    # 获取原方案
    old_prepare_plan = pd.read_csv(root + '/info/old_prepare_plan.csv', encoding='utf-8')
    old_prepare_plan['TYPE'] = old_prepare_plan['TYPE'].astype(str)
    # 获取型号和厂家信息
    producer_info = pd.read_csv(root + '/info/producer_info.csv', encoding='utf-8')
    devtype_info = pd.read_csv(root + '/info/devtype_info.csv', encoding='utf-8')
    devtype_info['DEV_TYPE'] = devtype_info['DEV_TYPE'].apply(lambda x: str(int(x)))
    producer_info['PRODUCER_CODE'] = producer_info['PRODUCER_CODE'].apply(lambda x: str(int(x)))
    # 数据库连接信息
    connection = pymysql.connect(**config)
    _cursor = connection.cursor()
    for index, row in data.iterrows():
        each_demand, to_db_demand_list = [], []
        time_range = np.vectorize(lambda s: pd.to_datetime(s).strftime('%Y-%m'))(pd.date_range(data.loc[index, 'start_month'], data.loc[index, 'end_month'], freq='MS')).tolist()
        try:
            # 生成月份-需求量字典
            start_index = time_range.index(start)
            end_index = time_range.index(end)
        except:
            print("时间超出范围")
            return None
        month_range = time_range[start_index:end_index + 1]
        # 若模型具有较好训练效果 则使用预测值 否则使用之前的方案
        if row['well_learnt'] == 1:
            predict_range = eval(row['predict_by_month'])[start_index:end_index + 1]
        else:
            old_plan_num = round(old_prepare_plan[old_prepare_plan['TYPE'] == row['dev_type']]['count'].values.tolist()[0] / center_num)
            predict_range = [old_plan_num for i in range(end_index - start_index + 1)]
        # 生成 型号-备件中心-月份-需求量字典
        if row['dev_type'] not in demand_dict:
            demand_dict[row['dev_type']] = {}
        month_apply_info = {}
        for i in range(end_index - start_index + 1):
            month_apply_info[month_range[i]] = round(predict_range[i])
            to_db_demand_list.append({'time': str(month_range[i]), 'demand': round(predict_range[i])})
        demand_dict[row['dev_type']][row['city']] = month_apply_info
        # 生成写入数据库的数据
        dev_info_split = row['dev_type'].split('_')
        type_name = str(devtype_info[devtype_info['DEV_TYPE'] == dev_info_split[1]]['NAME'].values.tolist()[0])
        producer_name = str(producer_info[producer_info['PRODUCER_CODE'] == dev_info_split[0]]['NAME'].values.tolist()[0])
        _cursor.execute('insert into t_smart_prepare(devtype, demand, dist_threshold, center_city, type_name, producer_name) values("'+ str(dev_info_split[0]) + "-" + str(dev_info_split[1]) + '","'+ str(to_db_demand_list).replace("'", "") + '",' + str(threshold) + ',"' + row["city"] + '","' + type_name + '","' + producer_name+'")')
    return demand_dict


# 根据room_day_demand.csv 获取start至end时间范围内的真实备件量
def get_true_demand(start, end):
    data = pd.read_csv(root + '/info/room_day_demand.csv', encoding='utf-8', usecols=['DEV_TYPE', 'PRODUCER_CODE', 'demand', 'MONTH', 'LOCATION'])
    data = data[(data['MONTH'] >= start) & (data['MONTH'] <= end)]
    data['TYPE'] = data.apply(lambda row: str(int(row['PRODUCER_CODE'])) + "_" + str(int(row['DEV_TYPE'])), axis=1)
    t_demand = data[['TYPE', 'MONTH', 'LOCATION', 'demand']].groupby(['TYPE', 'MONTH', 'LOCATION']).sum()
    t_demand = t_demand[t_demand['demand'] > 0]
    t_demand.reset_index(inplace=True)
    return t_demand


def find_nearest_stock_center(mapping, dist, stock_info, devtype, city, month):
    centers = list(set(mapping.values()))
    centers_with_dist = []
    for c in centers:
        dist_info = dist[(dist['start'] == city) & (dist['end'] == c)] if dist[(dist['end'] == city) & (dist['start'] == c)].empty else dist[(dist['end'] == city) & (dist['start'] == c)]
        if dist_info.empty:  # c == city 时
            centers_with_dist.append((c, 0))
        else:
            centers_with_dist.append((c, dist_info['time'].values.tolist()[0]))
    centers_with_dist = sorted(centers_with_dist, key=lambda x: x[1])
    for tu in centers_with_dist:
        if tu[0] in stock_info[devtype].keys() and stock_info[devtype][tu[0]][month] > 0:
            return tu[0], tu[1]
    return None, 'w'


def cal_plan_cost(t_month, dev_num, month_demand, dist, mapping, is_new_plan):
    center = '南昌市东湖区'
    m_cost, t_cost = '', ''
    # 设备管理费   m 表示单个设备的管理成本
    store_devnum = 0
    if mapping is None:
        for k in dev_num.keys():
            store_devnum += dev_num[k]
    else:
        t_month_split = list(map(lambda x: int(x), t_month.split("-")))
        last_month = str(t_month_split[0]) + '-' + str(t_month_split[1] - 1).zfill(2) if t_month_split[1] > 1 else str(t_month_split[0] - 1) + '-12'
        if last_month < prepare_start:
            last_month = None
        for k_type in dev_num.keys():
            for k_center in dev_num[k_type].keys():
                store_devnum += dev_num[k_type][k_center][t_month]
                if last_month is not None and dev_num[k_type][k_center][last_month] > 0:
                    dev_num[k_type][k_center][t_month] += dev_num[k_type][k_center][last_month]  # 本月库存 = 新增 + 上月库存剩余
    m_cost += str(int(store_devnum))+'m'

    # 更换的设备费用 c表示设备单价
    change_devnum = np.sum(month_demand['demand'])
    m_cost += '+' + str(int(change_devnum)) + 'c'

    # 运输成本  模拟故障-更换零件过程
    total_transport_tolls, total_transport_time, total_supply_time = 0, 0, 0
    for index, row in month_demand.iterrows():
        if mapping is not None:
            center = mapping[row['LOCATION']]
        # 记录每个零件的更换情况 包括：月份 型号 地点 预计供货中心 实际供货中心 是否临时补货 等待时间 成本 时间
        dev_change_record = pd.DataFrame()
        dev_change_record['month'] = [t_month]
        dev_change_record['location'] = [row['LOCATION']]
        dev_change_record['type'] = [row['TYPE']]
        dev_change_record['expect_center'] = [center]
        dev_change_record['is_supply'] = [0]
        dev_change_record['supply_time'] = [0]
        dev_change_record['is_new_plan'] = [is_new_plan]
        dev_change_record['demand'] = [row['demand']]

        # 计算运输的经济成本和时间成本
        dist_info = copy.deepcopy(dist[(dist['start'] == row['LOCATION']) & (dist['end'] == center)] if dist[(dist['start'] == center) & (dist['end'] == row['LOCATION'])].empty else dist[(dist['start'] == center) & (dist['end'] == row['LOCATION'])])
        dist_info.reset_index(inplace=True)
        if dist_info.empty:
            dist_info.loc[0, 'time'] = 0
            dist_info.loc[0, 'tolls'] = 0
        total_transport_time += dist_info['time'].values.tolist()[0]
        total_transport_tolls += round(dist_info['tolls'].values.tolist()[0] * 1.2)    # 运输费用是高速公路收费的线性函数(油费等其他开销)
        dev_change_record['time'] = dist_info['time'].values.tolist()[0]
        dev_change_record['tolls'] = round(dist_info['tolls'].values.tolist()[0] * 1.2)

        # 更新库存  型号-(备件中心)-(月份)-数量
        if mapping is None:
            dev_num[row['TYPE']] -= row['demand']
            if dev_num[row['TYPE']] < 0:
                # 仓库没有存货 需要临时补货
                total_supply_time += 1
                dev_change_record['is_supply'] = 1
                dev_change_record['supply_time'] = ['w']
            else:
                dev_change_record['real_center'] = center
        else:
            if dev_num[row['TYPE']][center][t_month] > 0:
                dev_num[row['TYPE']][center][t_month] -= row['demand']
                dev_change_record['real_center'] = center
            else:
                # 从离机房最近的有库存仓库提货 若均缺货则需临时补货
                option_center, option_time = find_nearest_stock_center(mapping, dist, dev_num, row['TYPE'], row['LOCATION'], t_month)
                if option_center is None:
                    total_supply_time += 1
                else:
                    total_transport_time += option_time
                dev_change_record['is_supply'] = 1
                dev_change_record['supply_time'] = option_time
                dev_change_record['real_center'] = option_center
        # 将每个更换零件的信息写入csv
        if not os.path.exists(root + '/info/dev_change_info.csv'):
            dev_change_record.to_csv(root + '/info/dev_change_info.csv', encoding='utf-8', header=True, index=False)
        else:
            dev_change_record.to_csv(root + '/info/dev_change_info.csv', encoding='utf-8', header=False, index=False, mode='a')
    # r表示比例系数，运输成本为高速公路收费的线性函数， 系数为r(>1)
    m_cost += '+' + str(round(total_transport_tolls, 2)) + 'r'
    t_cost += str(round(total_transport_time, 2)) + 'h'
    if total_supply_time > 0:
        t_cost += '+' + str(round(total_supply_time, 2)) + 'w'
    return m_cost, t_cost


def get_info_from_db(t_sql, columns, to_file):
    if not os.path.exists(root + '/info/' + to_file + '.csv'):
        connection = pymysql.connect(**config)
        try:
            df = pd.read_sql(t_sql, connection)
            df.drop_duplicates(inplace=True)
            df.dropna(inplace=True)
            df.columns = columns
            df.to_csv(root + '/info/' + to_file + '.csv', encoding='utf-8', header=True, index=False)
        finally:
            connection.close()


if __name__ == '__main__':
    root = os.path.abspath(os.path.dirname(__file__))
    prepare_start, prepare_end = '2017-10', '2017-12'
    province = 'jiangxi'
    config = {
        'host': "127.0.0.1",
        'port': 3306,
        'user': 'root',
        'password': 'root',
        'db': 'elec_server',
        'charset': 'utf8',
        'cursorclass': pymysql.cursors.DictCursor,
    }
    get_info_from_db("select PRODUCER_CODE, NAME, SN from t_pub_producer where PRODUCER_CODE IS NOT NULL and PRODUCER_CODE != '' ", ['PRODUCER_CODE', 'NAME', 'SN'], 'producer_info')
    get_info_from_db("select DEV_TYPE, NAME, RES_TYPE from t_pub_devtype where DEV_TYPE IS NOT NULL and DEV_TYPE != ''", ['DEV_TYPE', 'NAME', 'RES_TYPE'], 'devtype_info')

    # 获取真实备件需求量 城市间的距离等信息 备件中心映射关系
    center_mapping_info = pd.read_csv(root + '/info/'+ province +'_center_city_mapping.csv', encoding='utf-8')
    threshold = center_mapping_info['threshold'].unique().tolist()
    center_mapping_info = center_mapping_info[center_mapping_info['threshold'] == -1]
    center_mapping_info.set_index('LOCATION', inplace=True)
    center_city_mapping = center_mapping_info.to_dict()['CENTER_CITY']
    demand = get_true_demand(prepare_start, prepare_end)
    cost_dist = pd.read_csv(root + '/data/dist.csv', encoding='utf-8')

    # 获取原有方案的初始备件量 和 新方案的预测备件量
    old_plan_base, new_plan = gen_old_plan_base_demand(), gen_new_plan_demand(prepare_start, prepare_end, -1)
    # 计算方案的经济和时间成本
    # time_range = np.vectorize(lambda s: pd.to_datetime(s).strftime('%Y-%m'))(pd.date_range(prepare_start, prepare_end, freq='MS')).tolist()
    # for t in time_range:
    #     this_month_demand = demand[demand['MONTH'] == t]
    #     each_month_base = copy.deepcopy(old_plan_base)
    #     old_m_cost, old_t_cost = cal_plan_cost(t, each_month_base, this_month_demand, cost_dist, None, 0)
    #     new_m_cost, new_t_cost = cal_plan_cost(t, new_plan, this_month_demand, cost_dist, center_city_mapping, 1)
    #     compare = pd.DataFrame()
    #     compare['old_m_cost'] = [old_m_cost]
    #     compare['old_t_cost'] = [old_t_cost]
    #     compare['new_m_cost'] = [new_m_cost]
    #     compare['new_t_cost'] = [new_t_cost]
    #     compare['time'] = [t]
    #     compare['total_demand'] = [np.sum(this_month_demand['demand'])]
    #     if not os.path.exists(root + '/info/compare_info.csv'):
    #         compare.to_csv(root + '/info/compare_info.csv', encoding='utf-8', header=True, index=False)
    #     else:
    #         compare.to_csv(root + '/info/compare_info.csv', encoding='utf-8', header=False, index=False, mode='a')





