#-*- coding:utf-8 -*-
# 使用M-H算法利用Markov链进行采样
# 采样混合高斯分布，利用高斯分布当作转移概率密度函数

import numpy as np
import matplotlib.pyplot as plt

# numpy 生成任意维度的高斯分布
def numpy_gaussian(mu, sigma, nums):
    dims = mu.shape[0]
    # X = N0 * Sigma^0.5 + Mu
    return np.dot(np.random.randn(nums, dims), np.linalg.cholesky(sigma)) + mu

# 高斯分布输入x后的p(x|mu, sigma)
def single_gaussian_pdf(xs, mu, sigma):
    # @param xs：输入的数据点，m * n，当n=1维时，也要是(m, 1)形式的输入
    # @param mu：均值，一维数组
    # @param sigma：方差，二维数组
    m = xs.shape[0]        # 样本点个数
    pxs = np.zeros(m)      # 返回结果，m个概率值
    n = mu.shape[0]        # 高斯维度
    
    frac = ((2 * np.pi) ** (n/2)) * np.sqrt(np.linalg.det(np.mat(sigma)))
    # 逐个样本点求概率
    for i in range(m):
        ux = xs[i, :] - mu
        px = np.exp(-0.5 * np.dot(np.dot(ux.reshape(1, n), np.mat(sigma).I.A), ux.reshape(n, 1)))/frac
        pxs[i] = px
    return pxs


# 混合高斯分布的概率密度函数
def mix_gaussian_pdf(xs, alphas, mus, sigmas):
    #@param xs     : (m, n)的数据点，numpy.array， n = 1时也是二维数组 
    #@param alphas : 每个高斯成分的比例 length = K的列表
    #@param mus    : 每个高斯的均值，length = K的列表，列表元素是一维数组
    #@param sigmas : 每个高斯成分的方差，length = K的列表，列表元素是一个二维数组
    K = len(alphas)               # K个高斯成分
    m = xs.shape[0]               # 样本点个数
    
    pxs = np.zeros(m)
    for k in range(K):
        alpha = alphas[k]
        mu = mus[k]
        sigma = sigmas[k]
        pxs += alpha * single_gaussian_pdf(xs, mu, sigma)
    return pxs

# M-H采样
def mh_sampling():
    steps1 = 2000        # 先采样的点数
    steps2 = 8000        # 后面再采样的点数
    xn1 = 0.0        # 初始的点
    
    # 单个高斯分布（转移概率）的参数
    mu = np.array([0.0])
    sigma = np.array([[10.0]])
    
    # 混合高斯分布的参数
    alphas = [0.5, 0.27, 0.23]        # 权重
    mus = [np.array([0.0]), np.array([-3.0]), np.array([3.0])]           # 均值
    sigmas = [np.array([[1.0]]), np.array([[2.0]]), np.array([[0.5]])]   # 方差
    
    # 采样
    nums = []
    for i in range(steps1 + steps2):
        xn = numpy_gaussian(mu = mu, sigma = sigma, nums = 1)[0]                # 以q(xn|xn1)采样出来的点
        pxn1 = mix_gaussian_pdf(np.array([[xn1]]), alphas, mus, sigmas)[0]      # p(xn1)
        pxn = mix_gaussian_pdf(np.array([[xn]]), alphas, mus, sigmas)[0]        # p(xn)
        q_xn1_xn = single_gaussian_pdf(np.array([[xn1]]), mu, sigma)[0]         # q(xn1|xn)
        q_xn_xn1 = single_gaussian_pdf(np.array([[xn]]), mu, sigma)[0]          # q(xn|xn1)
        
        u = np.random.rand()
        if u <= min(1, (pxn * q_xn_xn1) / (pxn1 * q_xn1_xn)):
            xn1 = xn
        if i > steps1:
            nums.append(xn1)
    
    # 绘制对比图
    xs = np.linspace(-5, 5, 3000)
    mix_y = mix_gaussian_pdf(xs.reshape((-1, 1)), alphas = alphas, mus = mus, sigmas = sigmas)
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(xs, mix_y)
    plt.title("Mix Gaussian")
    plt.subplot(1, 2, 2)
    plt.hist(np.array(nums), bins = 200, normed = True)
    plt.title("M-H Sampling Mix Gaussian")
    plt.show()
    
def main():
    mh_sampling()
    
main()