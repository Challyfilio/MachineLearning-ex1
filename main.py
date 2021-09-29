import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 代价函数
def computeCost(x, y, theta):
    inner = np.power(((x * theta.T) - y), 2)
    result = np.sum(inner) / (2 * len(x))
    return result


# 梯度下降
def gradientDescent(x, y, theta, alpha, iters):  # alpha：学习率
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])  # 2列
    cost = np.zeros(iters)  # iters个0数组

    for i in range(iters):
        error = (x * theta.T) - y
        for j in range(parameters):
            term = np.multiply(error, x[:, j])  # (hθ(x)-y)x
            temp[0, j] = theta[0, j] - ((alpha / len(x)) * np.sum(term))  # 更新θ0，θ1
        theta = temp
        cost[i] = computeCost(x, y, theta)
    return theta, cost


# ex1data1 —————————————————————————————————————
'''
data = pd.read_csv('ex1data1.txt', header=None, names=['Population', 'Profit'])
# data.head()
# print(data.describe())
# data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))
# plt.show()

# 代价

# θT*x = θ0x0 + θ1x1 + θ2x2 + ... + θnxn， x0 = 1
data.insert(0, 'Ones', 1)
cols = data.shape[1]
x = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1:cols]
# print(x.head())
# print(y.head())
# print(data.describe())

# 转换x，y为矩阵，初始化theta
x = np.matrix(x.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0, 0]))
print(x.shape, y.shape, theta.shape)

# 梯度

alpha = 0.01  # learning rate
iters = 1000
θ, cost = gradientDescent(x, y, theta, alpha, iters)
print(cost)
print(computeCost(x, y, θ))

x = np.linspace(data.Population.min(), data.Population.max(), 100)
fx = θ[0, 0] + (θ[0, 1] * x)  # θ0+θ1x

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(data.Population, data.Profit, label='Training Data')
ax.plot(x, fx, 'r', label='Prediction')
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iters), cost)
ax.set_xlabel('Iters')
ax.set_ylabel('Cost')
plt.show()
'''

# ex1data2 —————————————————————————————————————
# '''
data2 = pd.read_csv('ex1data2.txt', header=None, names=['Size', 'Bedrooms', 'Price'])
data2 = (data2 - data2.mean()) / data2.std()  # 特征归一化
# print(data2.head())
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(data2.Size, data2.Bedrooms, data2.Price)
# ax.set_xlabel('Size')
# ax.set_ylabel('Bedrooms')
# ax.set_zlabel('Price')
# plt.show()

data2.insert(0, 'Ones', 1)
cols = data2.shape[1]
x2 = data2.iloc[:, 0:cols - 1]
y2 = data2.iloc[:, cols - 1:cols]

x2 = np.matrix(x2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0, 0, 0]))
print(x2.shape, y2.shape, theta2.shape)

alpha = 0.01
iters = 1000
θ2, cost2 = gradientDescent(x2, y2, theta2, alpha, iters)
print(computeCost(x2, y2, θ2))

x = np.linspace(data2.Size.min(), data2.Size.max(), 100)
y = np.linspace(data2.Bedrooms.min(), data2.Bedrooms.max(), 100)
fxy = θ2[0, 0] + (θ2[0, 1] * x) + (θ2[0, 2] * y)  # θ0+θ1x+θ2y

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data2.Size, data2.Bedrooms, data2.Price, 'r', label='Training Data')
ax.plot(x, y, fxy, 'r', label='Prediction')
ax.set_xlabel('Size')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('Price')
ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iters), cost2)
ax.set_xlabel('Iters')
ax.set_ylabel('Cost')
plt.show()
# '''
