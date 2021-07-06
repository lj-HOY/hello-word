# -*- coding: utf-8 -*-
"""
Created on Sun May 30 14:14:28 2021

@author: JIL

"""

'''载入数据'''
from sklearn import datasets

boston = datasets.load_boston()
x, y = boston.data, boston.target
'''引入标准化函数'''
from sklearn import preprocessing

x_MinMax = preprocessing.MinMaxScaler()
y_MinMax = preprocessing.MinMaxScaler()

''' 将 y 转换成 列 '''
import numpy as np

y = np.array(y).reshape(len(y), 1)
'''标准化'''
x = x_MinMax.fit_transform(x)
y = y_MinMax.fit_transform(y)

''' 按二八原则划分训练集和测试集 '''
from sklearn.model_selection import train_test_split

np.random.seed(2019)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

'''模型构建'''
from sklearn.neural_network import MLPRegressor

fit1 = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', alpha=0.01, max_iter=200)
# 第一个隐藏层有100个节点，第二层有50个，激活函数用relu，梯度下降方法用adam
# 惩罚系数为0.01，最大迭代次数为200
print ("fitting model right now")
fit1.fit(x_train, y_train)
pred1_train = fit1.predict(x_train)
'''计算训练集 MSE'''
from sklearn.metrics import mean_squared_error

mse_1 = mean_squared_error(pred1_train, y_train)
print ("Train ERROR = ", mse_1)
'''计算测试集mse'''
pred1_test = fit1.predict(x_test)
mse_2 = mean_squared_error(pred1_test, y_test)
print ("Test ERROR = ", mse_2)

'''结果可视化'''
import matplotlib.pyplot as plt

xx = range(0, len(y_test))
plt.figure(figsize=(8, 6))
#反归一化

#pred1_test = y_MinMax.inverse_transform(pred1_test)
y_test = y_MinMax.inverse_transform(y_test)
pred1_test = np.array(pred1_test).reshape(len(pred1_test),1)
pred1_test = y_MinMax.inverse_transform(pred1_test)

plt.plot(xx, pred1_test, 'ko-', label="Predict")
plt.plot(xx, y_test,  'ro-', label="Real")

# plt.scatter(xx, y_test, color="red", label="Sample Point", linewidth=3)
# plt.plot(xx, pred1_test, color="orange", label="Fitting Line", linewidth=2)
plt.legend()
plt.show()
#分析结果

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
y_t = []
y_h = []
for i in y_test:
    y_t.append(i)
for i in pred1_test:
    y_h.append(i)
print("MSE:", mean_squared_error(y_t, y_h))
print("MAS:", mean_absolute_error(y_t, y_h))
print("R2:", r2_score(y_t, y_h))
