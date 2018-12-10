import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':
    root = os.path.abspath(os.path.dirname(__file__))
    hours = [6.0, 12.0, 24.0, 36.0, 48.0]
    figure = plt.figure()
    for hour in hours:
        plt.ylim([0.5, 1.5])
        data = pd.read_csv(root + '/pinshan/result/' + str(hour) + 'h_best_pinshan_result.csv', usecols=["y_pred","y_true"])
        y_pred = list(map(lambda x: x+0.05 if x != 0 else np.nan, eval(data['y_pred'].values.tolist()[0])))
        y_true = list(map(lambda x: x if x != 0 else np.nan, eval(data['y_true'].values.tolist()[0])))
        # print(y_pred)
        # 一个月预测数据
        xdata = list(range(2160))
        plt.scatter(xdata, y_true[:2160], label='y_true')
        plt.scatter(xdata, y_pred[:2160], label='y_predict')
        plt.legend(loc='best')
        figure.savefig(root + '/pinshan/season_plot/'+ str(hour) + 'h_plot.png', transparent=True, dpi=80)
        figure.clear()