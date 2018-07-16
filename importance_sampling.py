#-*- coding:utf-8 -*-
# 使用Importence Sampling带有权重地采样，计算均值
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

# 寻找一个高斯分布可以完全在混合高斯分布上面
def draw_single_mix_gaussian():
    x = np.linspace(-5, 5, 3000)
    alphas = [0.5, 0.27, 0.23]        # 权重
    mus = [np.array([0.0]), np.array([-3.0]), np.array([3.0])]           # 均值
    sigmas = [np.array([[1.0]]), np.array([[2.0]]), np.array([[0.5]])]   # 方差
    
    mix_y = mix_gaussian_pdf(x.reshape((-1, 1)), alphas = alphas, mus = mus, sigmas = sigmas)
    
    single_y = 2.0 * single_gaussian_pdf(x.reshape((-1, 1)), mu = np.array([0.0]), sigma = np.array([[10.0]]))
    
    plt.figure()
    plt.plot(x, mix_y, 'b', x, single_y, 'r')
    plt.show()

# 重要性采样
def importance_sampling():
    # 对单独的高斯进行采样
    xs = numpy_gaussian(mu = np.array([0.0]), sigma = np.array([[10.0]]), nums = 5000)
    xs = np.sort(xs, axis = 0)    # 排序
    
    # 基于x生成single_gau和mix_gau对应的概率
    p1 = 2.0 * single_gaussian_pdf(xs.reshape((-1, 1)), mu = np.array([0.0]), sigma = np.array([[10.0]]))
    
    alphas = [0.5, 0.27, 0.23]        # 权重
    mus = [np.array([0.0]), np.array([-3.0]), np.array([3.0])]           # 均值
    sigmas = [np.array([[1.0]]), np.array([[2.0]]), np.array([[0.5]])]   # 方差
    p2 = mix_gaussian_pdf(xs.reshape((-1, 1)), alphas = alphas, mus = mus, sigmas = sigmas)
    
    # 权重
    xs_weights = p2 / p1
    
    # 计算均值
    mean_value = np.dot(xs.reshape(-1), xs_weights) / xs.shape[0]
    print("Mean value = ", mean_value)  # -0.0524097355004
    
def main():
    draw_single_mix_gaussian()
    importance_sampling()

main()