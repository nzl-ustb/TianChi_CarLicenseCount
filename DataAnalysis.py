#coding:utf-8

import pandas as pd
import matplotlib.pyplot as plt

##############导入数据
dir = 'D://python/nzl/TianChi/Car_Prediction/'
train = pd.read_table(dir + 'train.txt', engine='python')
test_A = pd.read_table(dir + 'test_A.txt', engine='python')
sample_A = pd.read_table(dir + 'sample_A.txt', engine='python')
print(train.info())
print(test_A.info())
print(train['day_of_week'].unique())
print(test_A['day_of_week'].unique())

##############找出目标值（即我们需要预测的）
#===== 箱型图，可查看异常值
plt.boxplot(train['cnt'])
plt.show()
#===== 分布图
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
from scipy import stats
from scipy.stats import norm, skew

sns.distplot(train['cnt'], fit=norm)
plt.show()

##############找出与目标变量最相关的量
#===== 脱敏时间变量
plt.plot(train['date'],train['cnt'])
plt.show()
#===== 日期
plt.scatter(train['day_of_week'], train['cnt'])
plt.show()
#===== 数字信息
print(train['cnt'].describe())
#===== 均方差MSE为评测函数，确定统计数据在评测函数中的指标
from sklearn.metrics import mean_squared_error
train['25%'] = 221
train['50%'] = 351
train['75%'] = 496
train['median'] = train['cnt'].median()
train['mean'] = train['cnt'].mean()
print(mean_squared_error(train['cnt'],train['25%']))
print(mean_squared_error(train['cnt'],train['50%']))
print(mean_squared_error(train['cnt'],train['75%']))
print(mean_squared_error(train['cnt'],train['median']))
print(mean_squared_error(train['cnt'],train['mean']))
## 可以大概看出来，由于存在异常点较多，导致统计量在时间轴上的表现并不是那么理想。现在还可以用的信息，只剩下了星期了，救命稻草之星期信息

##############开始对星期信息统计，分别分析周一周五的分布情况
for i in range(1, 8, 1):
    day_of_week = train[train['day_of_week'] == i]
    plt.subplot(2, 4, i)
    plt.plot(range(len(day_of_week)), day_of_week['cnt'])
plt.show()
## 明显可以把1-5和6,7分为两组去分析
#===== 简单分析一下按照星期的评测分数
res = train.groupby(['day_of_week'],as_index=False).cnt.mean()
xx = train.merge(res,on=['day_of_week'])
print(xx.head())
print(mean_squared_error(xx['cnt_x'],xx['cnt_y']))



