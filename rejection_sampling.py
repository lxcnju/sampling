#-*- coding:utf-8 -*-
# 使用Rejection Sampling拒绝采样，采样混合高斯分布
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

# 拒绝采样
def rejection_sampling():
    # 对单独的高斯进行采样
    xs = numpy_gaussian(mu = np.array([0.0]), sigma = np.array([[10.0]]), nums = 5000)
    xs = np.sort(xs, axis = 0)    # 排序
    
    # 基于x生成single_gau和mix_gau对应的概率
    p1 = 2.0 * single_gaussian_pdf(xs.reshape((-1, 1)), mu = np.array([0.0]), sigma = np.array([[10.0]]))
    
    alphas = [0.5, 0.27, 0.23]        # 权重
    mus = [np.array([0.0]), np.array([-3.0]), np.array([3.0])]           # 均值
    sigmas = [np.array([[1.0]]), np.array([[2.0]]), np.array([[0.5]])]   # 方差
    p2 = mix_gaussian_pdf(xs.reshape((-1, 1)), alphas = alphas, mus = mus, sigmas = sigmas)
    # 基于拒绝采样得到的混合高斯分布结果
    mix_gau_nums = []
    for i, x in enumerate(xs):
        u = np.random.rand()
        if u <= p2[i] / p1[i]:
            mix_gau_nums.append(x)
    # 打印出采样比例
    print("Rejection sampling ratio = ", len(mix_gau_nums)/xs.shape[0])    # 0.4986
    print("Mean value = ", sum(mix_gau_nums)/len(mix_gau_nums))            # -0.08131822
    # 绘制结果
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(xs, p2, 'b', xs, p1, 'r')
    plt.legend(["mix", "single"])
    plt.title("Single Mix Gaussian")
    plt.subplot(1, 2, 2)
    plt.hist(np.array(mix_gau_nums), bins = 200, normed = True)
    plt.title("Rejection Sampling")
    plt.show()

def main():
    rejection_sampling()

main()