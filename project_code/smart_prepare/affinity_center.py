from sklearn.cluster import AffinityPropagation
import pandas as pd
import os
import json
import numpy as np


def label_center_city(nodes, is_center, category, threshold):
    gen_file_name = path + '/info/' + province + '_center_city_mapping.csv'
    room_day_demand = pd.read_csv(path + '/info/room_day_demand.csv', encoding='utf-8')
    if 'CENTER_CITY' in room_day_demand.columns.tolist():
        room_day_demand.drop('CENTER_CITY', axis=1, inplace=True)

    data = pd.DataFrame()
    center_num = len(nodes[is_center])
    for k in range(center_num):
        member = category == k
        for city in nodes[member]:
            cur = pd.DataFrame()
            cur['LOCATION'] = [city]
            cur['CENTER_CITY'] = [nodes[is_center][k]]
            cur['threshold'] = [threshold]
            data = data.append(cur)
    if not os.path.exists(gen_file_name):
        data.to_csv(gen_file_name, encoding='UTF-8', header=True, index=False)
    else:
        data.to_csv(gen_file_name, encoding='UTF-8', header=False, index=False, mode='a')
    df = pd.merge(room_day_demand, data, on=['LOCATION'], how='left')
    df.to_csv(path + '/info/room_day_demand.csv', encoding='utf-8', header=True, index=False)


def update_geo_json(threshold):
    index = np.array(range(len(cities)))
    geo_json = json.load(open('data/geo' +'.json', 'r'))
    for k in range(len(center)):
        member = labels == k
        for item in index[member]:
            geo_json[item]['is_center'] = False
            geo_json[item]['label'] = k
    label = 0
    for i in center:
        geo_json[i]['is_center'] = True
        geo_json[i]['label'] = label
        label += 1
    if not os.path.exists('data/geo/'):
        os.mkdir('data/geo/')
    file = open('data/geo/geo_' + str(threshold) + '.js', 'w')
    file.write('var cities=' + json.dumps(geo_json) + ";")
    file.close()


def distance_to_likelihood(dist):
    feature_dist = json.load(open(path + '/data/' + province + '_dist_feature.json', 'r'))
    mmax = feature_dist['max_dist']
    return 1 - dist/mmax


def affinity_center(likelihood_matrix, pref=-0.5):
    af = AffinityPropagation(preference=pref, affinity='precomputed').fit(likelihood_matrix)
    return af.cluster_centers_indices_, af.labels_


def find_farthest_distance(is_center, category, nodes):
    center_num = len(nodes[is_center])
    index = np.array(range(len(nodes)))
    center_index = index[is_center]
    distance = 1
    for k in range(center_num):
        member = category == k
        for city_index in index[member]:
            if X[center_index[k]][city_index] < distance:
                # print(nodes[city_index], nodes[center_index[k]], X[center_index[k]][city_index])
                distance = X[center_index[k]][city_index]
    return distance


if __name__ == '__main__':
    path = os.path.abspath(os.path.dirname(__file__))
    province = 'jiangxi'
    X = pd.read_csv('data/dist_weight_matrix.csv', encoding='utf-8').values
    cities = pd.read_table(path + '/data/jiangxi_city.txt', header=None, encoding='GBK', delim_whitespace=True).values[0]

    # 传播聚类计算出中心
    thresholds = [-1, 100, 150, 200, 250]
    gen_file_name = path + '/info/' + province + '_center_city_mapping.csv'
    if os.path.exists(gen_file_name):
        os.remove(gen_file_name)
    for threshold in thresholds:
        if threshold == -1:
            center, labels = affinity_center(X)
        else:
            threshold_mat_value = distance_to_likelihood(threshold)
            i = -1
            while True:
                center, labels = affinity_center(X, i)
                # 选址中心
                if find_farthest_distance(center, labels, cities) < threshold_mat_value:
                    i += 0.01
                else:
                    centers = cities[center]
                    break
        # 为每个城市标注所属中心
        label_center_city(cities, center, labels, threshold)
        update_geo_json(threshold)


