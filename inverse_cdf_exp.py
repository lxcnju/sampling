#-*- coding:utf-8 -*-
# 使用Inverse CDF Sampling，概率累积函数逆变换采样
# 以指数分布为例y = lambda * exp(-lambda * x)

import numpy as np
import matplotlib.pyplot as plt

lamd = 1.0             # lambda系数
shape = (5000, )       # 数据点大小
bins = 200             # 绘制直方图的bins

# 指数分布的概率累积函数的逆变换，x = -ln(1 - y)/lambda
def inverse_cdf(lamd, shape):
    uni_nums = np.random.uniform(0.0, 1.0, shape)
    uni_nums[uni_nums == 1.0] = 0.99999
    exp_nums = -np.log(1 - uni_nums)/lamd
    return exp_nums

# 采样并绘制直方图
def sampling_exp():
    np_exp_nums = np.random.exponential(scale = 1.0/lamd, size = shape)   # numpy的
    exp_nums = inverse_cdf(lamd = lamd, shape = shape)     # 自己实现的
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.hist(np_exp_nums, bins = bins, normed = True)
    plt.title("numpy exponential")
    plt.subplot(1, 2, 2)
    plt.hist(exp_nums, bins = bins, normed = True)
    plt.title("my exponential")
    plt.show()


def main():
    sampling_exp()


main()
