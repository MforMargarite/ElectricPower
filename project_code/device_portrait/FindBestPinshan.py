"""
    Topic:寻找设备最适合的频闪时间，考察范围为（1——48小时）
    Date:2018-06-09
"""
import pymysql as db  # 数据库包
import pymysql.cursors
import pandas as pd
from sklearn.model_selection import GridSearchCV
import os
import numpy as np
from util import *
from Config import Config
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn import preprocessing


def GenerateAlarmWithProducer(province, filepath):
    """
    生成拥有生产厂家的Alarm文件
    :return:
    """
    #################配置#########################
    # 省份
    # csv保存路径
    path = filepath + province + '_electric'
    # 数据库配置
    config = {
        'host': "172.16.135.6",
        'port': 3306,
        'user': 'root',
        'password': '10086',
        'db': province,
        'charset': 'utf8',
        'cursorclass': pymysql.cursors.DictCursor,
    }
    connection = pymysql.connect(**config)

    try:
        sql3 = "select NE_OBJ_ID,DATE_FORMAT(FROM_UNIXTIME(ALARM_EMS_TIME/1000),'%Y-%m-%d') " \
               ", ALARM_EMS_TIME, ALARM_EMS_RESUME_TIME, ALARM_CAUSE,PRODUCER_NAME,DEV_TYPE_ID from t_rt_history_alarm where ALARM_CAUSE in" \
               " ('trafficReplaceableUnitProblem','coolingSystemFailure','configurationOrCustomisationError','cntrlBusFail'," \
               "'trafficReplaceableUnitTypeMismatch','internalCommFail','coolingFanFailure','replaceableUnitProblem'," \
               "'internalFail','diskFailure','selfTestFail','rectifierHighVoltage','NE_NOT_LOGIN','NE_COMMU_BREAK'," \
               "'backplaneFailure','laserTxPowerThresholdHigh','txLaserDegraded','trafficReplaceableUnitUnknown'," \
               "'fuseFailure','FAN_FAIL','Fans Major','Card/Slot Fail','memoryProblem','powerSupplyFailure','powerProblem'," \
               "'tempOutOfRange','TEMP_ALARM','dccFail','lossOfSignal','farEndReceiverFailure','lossOfFrame','IN_PWR_ABN'," \
               "'Degraded Signal','Loss Of Frame','Loss Of Signal','LOF','Temperature Out Of Range','High Temperature'," \
               "'Sfwr Environment Problem','HouseKeeping','Housekeeping Alarm','And Battery Failure','Battery Failure'," \
               "'PWRSUSP','PWR','Power Failure','VOLTAGELOW','Inside Failure','Underlying Resource Unavailable'," \
               "'Replaceable Unit Problem','MIB backup misaligned','Internal Communication Problem'," \
               "'Replaceable Unit Missing','Transport Failure','Transport Outgoing Failure','Fuse Failure'," \
               "'Transmitter Failure','Equipment Failure','Cooling Fan Failure','PWRADJFAIL','NE not reachable by Control Plane'," \
               "'FANSPEEDHIGH','Pump Failure',' LossOfSignal','fanFailure',' PowerProblem','RSM_配置采集单元（探针）异常'," \
               "' TransmitterFailure','TransmissionError',' fanDegraded','BackupFailed','fanDegraded','Loss Of Signal'," \
               "'Loss Of Frame','Equipment power failure at connector B','Equipment power failure at connector A'," \
               "'Equipment power failure at return connector A','Equipment Failure','Fan Failure','Equipment Low Tx power'," \
               "'Optical Amplifier Gain Degrade Low','LOF','TempHighAlarm','PKG_FAIL','FAN_FAIL','BUS_ERROR','LAPS_FAIL'," \
               "'MEM_FAIL','FCS_ERR','R_LOS','R_LOF','IN_PWR_ABN','IN_PWR_LOW','光模块的接收功率过低','TEMP_OVER'," \
               "'PWR_MAJ_ALM','POWER_FAIL','TEMP_ALARM','POWER_ABNORMAL','POWERALM','电源整体功能失效','NE_NOT_LOGIN'," \
               "'RSM_采集北向接口故障','NE_COMMU_BREAK','FAN_FAIL','BD_STATUS','SYN_BAD','R_OOF','COMMUN_FAIL','HARD_BAD'" \
               ",'SYSBUS_FAIL','BIOS_STATUS','BUS_ERR','GNE_CONNECT_FAIL','已配置的单板不能通信','单板检测到电源故障'," \
               "'LSR_WILL_DIE','OUT_PWR_ABN','SUBCARD_ABN','BD_NOT_INSTALLED','CHIP_ABN','PATCH_ERR','LSR_NO_FITED'," \
               "'NE_DATA_INCONSISTENCY','IN_PWR_FAIL','ABNORMAL_PROCESS_STATE','已配置的单板可以通信，但上报单板故障'," \
               "'输出板的2M输入均不可用，可能时钟板或LCIM的输出有问题','输出板的E1输入均不可用，可能时钟板或LCIM的输出有问题'," \
               "'输出板的10M输入均不可用，可能时钟板或LCIM的输出有问题','TODI单板检测不到该DCLS输入源','SYNC_FAIL'," \
               "'IN_PWR_HIGH','LASER_MOD_ERR','CHIP_FAIL','数据库需要维护','INPWR_FAIL','光模块拔出'," \
               "'请检查天线、馈线等硬件连接，并确认天线周围有无强干扰','光模块整体功能失效','CFCARD_FAILED','防尘网清洗告警'," \
               "'光模块异常','风扇整体功能失效','光模块局部功能失效','单板电压超过致命阈值','光模块的接收功率过高'," \
               "'子卡整体功能失效','时钟板本振故障。','Signal Degrade','Loss Of Signal','RDI','Power Failure - B'," \
               "'ME loss of communication','Circuit Pack Missing','Loss Of Shelf Sec. Timing Ref.'," \
               "'Auto STS3C Path Switch Complete-Sig. Fail','Database Save and Restore Failed','NE_Backup_Failed','Comms fail alarm'," \
               "'Circuit Pack Failed','Qecc-Comms_Fail','Duplicate Shelf Detected','OS-Optical_Power_High','Fan Failed'," \
               "'Internal Mgmt Comms Suspected','Circuit Pack Missing - Pluggable','Client Service Mismatch'," \
               "'Circuit Pack Upgrade Failed','I/O Module Missing','Circuit Pack Failed - Sync','Circuit Pack Mismatch'," \
               "'Circuit Pack Mismatch - Pluggable','Bipolar Violations','Corrupt Inventory Data','lossOfSignal'," \
               "'STM16光物理接口 信号丢失(LOS)','激光器温度越限','探测点温度越限','detectTemp','LsrTemp-TCA'," \
               "'探测点温度(℃)越限','-48V电源分配箱1高压告警','-48V电源分配箱2无输入','-48V电源输入Sci无输入','powerSciLow'," \
               "'-48V电源输入Sci低压告警','powerSciNoInput','powerBox2Port2NoInput','powerQxiHigh','powerSciHigh'," \
               "'powerQxiNoInput','powerBox1NoOutput','-48V电源输入Sci高压告警','-48V电源输入Qxi高压告警'," \
               "'-48V电源分配箱1低压告警','-48V电源分配箱1无输入','-48V电源分配箱2低压告警','电源故障','-48V电源分配箱2高压告警'," \
               "'-48V电源输入Qxi无输入','-48V电源输入Qxi低压告警','powerProblem','powerBAccessFailure','powerAAccessFailure'," \
               "'B路电源接入故障','A路电源接入故障','环境 -48V电源分配箱2无输入','coolingFanFailure','风扇故障'," \
               "'STM1光物理接口 光模块未认证','fanMiss','应安板未安装','单板运行不正常','晶振老化或者时钟参考源频率越界'," \
               "'风扇故障2','激光器偏流越限','UNKNOWN_CARD','单板软件运行不正常','单板 板类型未知','PLUGGABLE_TRANSCEIVER_DISMOUNT'," \
               "'单板 晶振老化或者时钟参考源频率越界','ABSENCE_WARNING','背板总线错','单板 应安板未安装','单板 板类型失配'," \
               "'SHELF_ABSENCE','单板 电源故障','环境 风扇2故障','环境 风扇故障','LOS','PWR_FAIL','PWRL_NFB_B','FAN'," \
               "'Association Failed','FAN_FAIL','PKG_FAIL','COMM_FAIL','BUS_FAIL','CONTBUS_RMV','CPU_FAIL','LOS','Service Degraded'," \
               "'IN_PWR_ABN','RX-oPower-L-Warning','RX-oPower-Low','Optical Power High Back Reflection,[sub_slot=2]'," \
               "'High Temperature','Temperature-TC','TEMP_ALARM','High Temperature Pump','TEMP_OVER','Low Temperature'," \
               "'High Temperature,[sub_slot=33]','ENV-Temp-High','MCP-Start-Fail','Power Failure','POWER_ABNORMAL'," \
               "'POWER_FAIL','DC-IN-Fail','Input Voltage 1 Failure','Air Flow','High Tx Power','Service Failure','Disconnection'," \
               "'Low Tx Power','Laser Bias High','AN Failure','BD_STATUS','FAN_FAIL','COMMUN_FAIL','NE_COMMU_BREAK'," \
               "'NE_NOT_LOGIN','Fan-FAIL','Programming Fault','Communication Failure','Low Tx Power ,[sub_slot=33]','Card Failure'," \
               "'Card-Fail') and ALARM_EMS_TIME >= 1451577600000 " \
               "and ALARM_EMS_TIME < 1525104000000 and NE_OBJ_ID not in ('') and ALARM_EMS_TIME <= ALARM_EMS_RESUME_TIME"
                # 数据时间维度为：2016-01-01 00:00:00 到 2018-05-01 00:00:00
        dff2 = pd.read_sql(sql3, connection)
        dff2 = dff2.drop_duplicates()
        names = ['NE_OBJ_ID', 'ALARM_EMS_TIME_DAY', 'ALARM_EMS_TIME', 'ALARM_EMS_RESUME_TIME', 'ALARM_CAUSE','PRODUCER_NAME','DEV_TYPE_ID']
        dff2.columns = names
        print("alarm文件开始写入")
        dff2.to_csv(path + "_alarm.csv", index=False, header=True)
        # 从alarm文件生成data_drop文件
        # 读取初始告警表
        df1 = pd.read_csv(path + "_alarm.csv", encoding='gbk', sep=',')
        df1 = df1.sort_values(by=['NE_OBJ_ID', 'ALARM_EMS_TIME']).reset_index(drop=True)
        a = pd.Series()
        # 首先合并同一时间发生的故障，故障恢复时间取最大值
        for i in range(df1.shape[0]):
            if i > 0:
                id = df1.loc[i, 'NE_OBJ_ID']
                if str(id) in str(df1.loc[(i - 1), 'NE_OBJ_ID']):
                    m = df1.loc[i, 'ALARM_EMS_TIME'] - df1.loc[(i - 1), 'ALARM_EMS_TIME']
                    if m == 0:
                        a = a.append(pd.Series([(i - 1)]))
                        df1.loc[i, 'ALARM_EMS_RESUME_TIME'] = max(df1.loc[(i - 1), 'ALARM_EMS_RESUME_TIME'],
                                                                  df1.loc[i, 'ALARM_EMS_RESUME_TIME'])
                        print(i - 1)
        a = a.drop_duplicates()
        df1 = df1.drop(a, axis=0).reset_index(drop=True)
        print("data_drop文件开始写入")
        df1.to_csv(path + "_data_drop.csv", index=False, header=True)
    finally:
        connection.close()


def get_err_date(x):
    result = []
    for index, row in x.iterrows():
        begin = datetime.datetime.strptime(row['ALARM_EMS_TIME'], "%Y-%m-%d")
        end = datetime.datetime.strptime(row['ALARM_EMS_RESUME_TIME'], "%Y-%m-%d")
        while begin <= end:
            result.append(begin)
            begin += datetime.timedelta(days=1)
    return list(set(result))


def gen_feature_data(filePath, province, hour, config):
    """ 生成数据：设备id，日期，故障率(该日期是否发生故障或在故障期内）"""
    try:
        df = pd.read_csv(filePath + province + '_electric' + str(hour) + 'h_data_clean.csv', encoding='utf-8', sep=',')
    except:
        df = pd.read_csv(filePath + province + '_electric' + str(hour) + 'h_data_clean.csv', encoding='gbk', sep=',')
    df['ALARM_EMS_TIME'] = df.ALARM_EMS_TIME.apply(timestamp_to_day)
    df['ALARM_EMS_RESUME_TIME'] = df.ALARM_EMS_RESUME_TIME.apply(timestamp_to_day)
    df = df[~df['ALARM_CAUSE'].isin(['Loss Of Signal', 'LOF', 'RDI'])]
    df = df.loc[:, ['NE_OBJ_ID', 'ALARM_EMS_TIME', 'ALARM_EMS_RESUME_TIME']]

    data = pd.DataFrame(columns=['ALARM_EMS_TIME', 'NE_OBJ_ID', 'result','date','dow','doy','day','month','t_m24','t_m_week'])
    data.to_csv(filePath + province + '_' + str(hour) + '_feature.csv', index=False, header=True, encoding='utf-8')
    c = df.groupby(["NE_OBJ_ID"], as_index=False).size()
    c = pd.DataFrame(c, columns=["COUNT"]).reset_index()
    for index, rows in c.iterrows():
        if rows["COUNT"] >= 10:
            tem = pd.DataFrame()
            tem['ALARM_EMS_TIME'] = pd.date_range(config.start_time, config.end_time)        #   时间范围
            tem['NE_OBJ_ID'] = rows["NE_OBJ_ID"]
            err_date = get_err_date(df[df['NE_OBJ_ID'] == rows["NE_OBJ_ID"]])
            tem['result'] = tem.ALARM_EMS_TIME.apply(lambda x: 1 if x in err_date else 0)
            tem['date'] = tem.ALARM_EMS_TIME.apply(lambda x: pd.to_datetime(x))
            tem['dow'] = tem.date.apply(lambda x: x.dayofweek)
            tem['doy'] = tem.date.apply(lambda x: x.dayofyear)
            tem['day'] = tem.date.apply(lambda x: x.day)
            tem['month'] = tem.date.apply(lambda x: x.month)
            tem['t_m24'] = tem.date.apply(get_prev_days, args=(1, tem))
            tem['t_m_week'] = tem.date.apply(get_prev_days, args=(7, tem))
            data = data.append(tem, ignore_index=True)
        if data.shape[0] > 100000:
            data.to_csv(filePath + province + '_' + str(hour) + '_feature.csv', index=False, header=False, encoding='utf-8', mode='a')
            data = pd.DataFrame(columns=['ALARM_EMS_TIME', 'NE_OBJ_ID', 'result', 'date', 'dow', 'doy', 'day', 'month', 't_m24', 't_m_week'])
    data.to_csv(filePath + province + '_' + str(hour) + '_feature.csv', index=False, header=False, encoding='utf-8', mode='a')


def normalize(matrix):
    min_max = preprocessing.MinMaxScaler()
    return min_max.fit_transform(matrix)


def cos_distance_matrix(X):
    # 余弦计算时间序列相似度
    n = X.shape[0]
    matrix = np.ones((n, n))
    for index, row in X.iterrows():
        for other in range(index+1, n):
            x = row['result']
            y = X.loc[other, 'result']
            like = np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))
            matrix[index][other] = like
            matrix[other][index] = like
    matrix = normalize(matrix)
    print(matrix)
    return matrix


def labelize(filePath, province, hour):
    data = pd.read_csv(filePath + province + '_' + str(hour) + '_feature.csv', encoding='utf-8')
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(data['NE_OBJ_ID'].unique())
    data['ID'] = label_encoder.transform(data['NE_OBJ_ID'])
    data.to_csv(filePath + province + '_' + str(hour) + '_feature.csv', header=True, index=False, encoding='utf-8')


def get_predict_model(filePath, province, hour):
    col = ['dow', 'doy', 'day', 'month', 't_m24', 't_m_week', 'ID']
    sample = pd.read_csv(filePath + province + '_' + str(hour) + '_feature.csv', encoding='utf-8', low_memory=False)
    sample.dropna(inplace=True)
    sample.sort_values(by=['ALARM_EMS_TIME'], ascending=True, inplace=True)
    sample.reset_index(inplace=True)
    n = sample.shape[0]
    train_size = int(0.7 * n)+1
    trainX = sample.loc[:train_size, col]
    train_y = sample.loc[:train_size, 'result']
    testX = sample.loc[train_size:, col]
    test_y = sample.loc[train_size:, 'result']
    # train_test_split(sample[col], sample['result'], test_size=0.3, random_state=80, stratify=sample['result'])
    param_grid = {
        'n_estimators': range(2000, 4001, 500),
        'min_samples_split': range(60, 121, 20)
    }
    estimator = GradientBoostingClassifier(subsample=0.8, random_state=80, max_depth=5, min_samples_leaf=5)
    cv = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=80)
    gbm = GridSearchCV(estimator=estimator, param_grid=param_grid, refit=True, n_jobs=-1, return_train_score=True, scoring='roc_auc', cv=cv)
    # 训练和输出
    gbm.fit(trainX, train_y)
    # 预测
    y_pred = gbm.predict(testX)
    y_predprob = gbm.predict_proba(testX)[:, 1]
    # 输出结果至csv
    clf = gbm.best_estimator_
    result = pd.DataFrame()
    result["PINSHAN_TIME"] = [hour]
    result["BEST_ESTIMATOR"] = [str(gbm.best_estimator_)]
    result["FEATURE_RANK"] = [str(clf.feature_importances_)]
    result["TRAIN_ROC_AUC"] = [str(gbm.best_score_)]
    result["ACCURACY"] = [metrics.accuracy_score(test_y.values, y_pred)]
    result["ROC_AUC"] = [metrics.roc_auc_score(test_y, y_predprob, average='weighted')]
    result["y_pred"] = [str(y_pred)]
    result["y_true"] = [str(test_y)]
    result.to_csv(root + '/pinshan/result/' + str(hour) + 'h_best_pinshan_result.csv', header=True, index=False,
                  encoding='utf-8')
    print(str(hour) + "h下roc_auc最佳得分为：" + str(metrics.roc_auc_score(test_y, y_predprob, average='weighted')))
    # 输出模型
    joblib.dump(clf, root + '/pinshan/result/' + str(hour) + '_GDBT.model')


# 获得最佳频闪
# 数据从2016/1/3至2018/4/20
if __name__ == '__main__':
    root = os.path.abspath(os.path.dirname(__file__))
    filePath = root + '/pinshan/'
    province = 'jiangxi'
    config = Config()
    time = [6.0, 12.0, 24.0, 36.0, 48.0]
    for i in time[::-1]:
        print(i)
        # if not os.path.exists(filePath + province + '_' + str(i) + '_feature.csv'):
        #     gen_feature_data(filePath, province, i, config)
        #     labelize(filePath, province, i)
        get_predict_model(filePath, province, i)
