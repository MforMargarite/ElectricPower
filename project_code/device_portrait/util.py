import time,datetime

import pandas as pd


def string_to_timestamp(s): # 时间字符串转时间戳
    timeArray = time.strptime(s, '%Y-%m-%d %H:%M:%S')
    timestamp = int(time.mktime(timeArray))
    return timestamp


def timestamp_to_date(timeStamp):
    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    dateArray = datetime.datetime.utcfromtimestamp(timeStamp)
    # otherStyleTime2 = dateArray.strftime("%Y--%m--%d %H:%M:%S")
    return otherStyleTime,dateArray


def timestamp_to_day(timeStamp):
    timeStamp /= 1000
    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y-%m-%d", timeArray)
    return otherStyleTime


def get_prev_days(x,n_days,data,pday=pd.Timedelta('1 day')):
    try:
        re = data[data.date == x-n_days*pday].result.values[0]
    except:
        re = data[data.date == x].result.values[0]
    return re
