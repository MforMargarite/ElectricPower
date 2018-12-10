import json
import urllib.request as request
import urllib.parse as parse
import pandas as pd
import os
import numpy as np
import joblib
from sklearn.preprocessing import MaxAbsScaler


# 获取所有机房所在城市集合
def get_machine_room_city(prov):
    if not os.path.exists(root + '/data/' + prov + '_city.txt'):
        cities = pd.read_csv(root + '/info/t_spc_room.csv', encoding='GBK', usecols=['LOCATION'])
        cl = cities['LOCATION'].unique().tolist()
        cl.sort()
        cdata = ''
        for c in cl:
            cdata += c + ' '
        print(cdata, len(cl))
        to_txt = open(root + '/data/' + prov + '_city.txt', 'w')
        to_txt.write(cdata[:-1])
        to_txt.flush()
        to_txt.close()


def get_geo(city):
    geo_url = "http://restapi.amap.com/v3/geocode/geo?key=" + key[0] + "&address=" + parse.quote(city)
    res = json.loads(request.urlopen(request.Request(geo_url)).read(), encoding='utf-8')
    geo_info = res['geocodes'][0]['location']
    return geo_info


def gen_distance(path):
    cities = pd.read_table(path + '/data/jiangxi_city.txt', header=None, encoding='GBK', delim_whitespace=True).values[0]
    n = len(cities)
    geo_dict = dict()
    cur_index, key_index = 0, 0

    for i in range(0, n - 1):
        start = cities[i]
        if start in geo_dict:
            ori = geo_dict[start]
        else:
            ori = get_geo(start)
            geo_dict[start] = ori
        each_city = pd.DataFrame()
        for j in range(i + 1, n):
            end = cities[j]
            if end in geo_dict:
                dest = geo_dict[end]
            else:
                dest = get_geo(end)
                geo_dict[end] = dest
            print(start, end)
            if cur_index == 1990:
                key_index += 1
                cur_index = 0
            url = 'https://restapi.amap.com/v3/direction/driving?key=' + key[key_index] + '&origin=' + ori + '&destination=' + dest + '&output=json'
            response = json.loads(request.urlopen(url).read())
            dist_info = response['route']['paths'][0]
            cur = pd.DataFrame()
            cur['start'] = [start]
            cur['end'] = [end]
            cur['origin'] = [ori]
            cur['dest'] = [dest]
            cur['distance'] = int(dist_info['distance']) / 1000  # 公里
            cur['time'] = int(dist_info['duration']) / 3600  # 小时
            cur['tolls'] = float(dist_info['tolls'])    # 收费
            each_city = each_city.append(cur)
            cur_index += 1
        if not os.path.exists(path + '/data/dist.csv'):
            each_city.to_csv(path + '/data/dist.csv', header=True, index=False, encoding='utf-8')
        else:
            each_city.to_csv(path + '/data/dist.csv', header=False, index=False, encoding='utf-8', mode='a')


def min_max_map(matrix):
    mmax = np.max(matrix)
    mmin = np.min(matrix)
    feature_dict = {'max_dist': mmax}
    features = open(root + '/data/' + province + '_dist_feature.json', 'w')
    features.write(json.dumps(feature_dict))
    features.flush()
    features.close()
    return [1-(i-mmin)/(mmax - mmin) for i in matrix]


def gen_dist_matrix(path, filename):
    info = pd.read_csv(path + filename, encoding='utf-8')
    cities = pd.read_table(path + '/data/jiangxi_city.txt', header=None, encoding='GBK', delim_whitespace=True).values[0]
    n = len(cities)
    dist_matr = pd.DataFrame(columns=cities, index=range(n))
    matrix = np.zeros((n,n))
    row = 0
    for c in cities:
        c_dist = [0]+info[info['start'] == c]['distance'].values.tolist()
        dist_matr.iloc[row:, row] = c_dist
        dist_matr.iloc[row, row:] = c_dist
        matrix[row:, row] = c_dist
        matrix[row, row:] = c_dist
        row += 1
    # 将距离映射到[0,1]
    matrix = min_max_map(matrix)
    result = pd.DataFrame(matrix, columns=cities)
    # 距离矩阵
    dist_matr.to_csv(path + '/data/dist_matrix.csv', header=True,index=False,encoding='utf-8')
    # 距离权重矩阵
    result.to_csv(path + '/data/dist_weight_matrix.csv', header=True, index=False, encoding='utf-8')


def to_json(path, filename):
    data = pd.read_csv(path + filename, encoding='utf-8')
    geo_info = []
    cities = data['start'].unique().tolist()
    for c in cities:
        origin = data[data['start'] == c].reset_index().loc[0, 'origin']
        geo_info.append({'city': c, 'center': origin})
    last = data.shape[0]-1
    geo_info.append({'city': data.loc[last, 'end'], 'center': data.loc[last, 'dest']})
    geo_json = json.dumps(geo_info)
    json_file = open('data/geo.json', 'w')
    json_file.write(geo_json)
    json_file.close()


if __name__ == '__main__':
    root = os.path.abspath(os.path.dirname(__file__))
    key = ['751ccbfd9d9cde3153adac28b17f6eae', '467d86c9daff20f48e29ea33d9c857bc', '2fa2700966ecffc046a628f3ce45cdf9', '70f68bb14ea4c59829e2e69c66c6bfd8', 'c5e3dcb788e91bb52b1569cfddcf90e5','8a5583792fd53c8ac1e75be438479564', '8fee15cb1a652dec6ca29d3357286216']
    province = 'jiangxi'
    # get_machine_room_city(province)
    # 从高德API获取江西省县级以上城市间的驾车距离和时间
    gen_distance(root)
    # 由距离信息生成矩阵Dij
    # gen_dist_matrix(root, '/data/dist.csv')
    # to_json(root, '/data/dist.csv')