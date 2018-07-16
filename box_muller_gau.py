#-*- coding:utf-8 -*-
# ʹ��Box-Muller������˹�ֲ�
# һά�Ͷ�ά�ĸ�˹

import numpy as np
import matplotlib.pyplot as plt

# numpy ��������ά�ȵĸ�˹�ֲ�
def numpy_gaussian(mu, sigma, nums):
    dims = mu.shape[0]
    # X = N0 * Sigma^0.5 + Mu
    return np.dot(np.random.randn(nums, dims), np.linalg.cholesky(sigma)) + mu


# ����Box-Muller���ɶ�ά��˹�ֲ�
def box_muller(mu, sigma, nums):
    dims = mu.shape[0]
    norm_gau_nums = []
    # ������dims��һά�ĸ�˹�ֲ������Կ������໥������
    for d in range(dims):
        uni_nums_1 = np.random.uniform(0.0, 1.0, (nums,))
        uni_nums_2 = np.random.uniform(0.0, 1.0, (nums,))
        norm_gau_nums_one_dim = np.sqrt(-2 * np.log(uni_nums_1)) * np.cos(2 * np.pi * uni_nums_2)
        norm_gau_nums.append(norm_gau_nums_one_dim)
    
    return np.dot(np.array(norm_gau_nums).reshape(-1, dims), np.linalg.cholesky(sigma)) + mu


# һά��
def draw_one_dim_gau():
    np_gau = numpy_gaussian(mu = np.array([1.0]), sigma = np.array([[2.0]]), nums = 5000)
    my_gau = box_muller(mu = np.array([1.0]), sigma = np.array([[2.0]]), nums = 5000)
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.hist(np_gau, bins = 200, normed = True)
    plt.title("numpy gaussian one dim")
    plt.subplot(1, 2, 2)
    plt.hist(my_gau, bins = 200, normed = True)
    plt.title("my gaussian one dim")
    plt.show()

# ��ά��
def draw_two_dim_gau():
    np_gau = numpy_gaussian(mu = np.array([-1.0, 1.0]), sigma = np.array([[1.0, 0.5], [0.5, 4.25]]), nums = 5000)
    my_gau = box_muller(mu = np.array([-1.0, 1.0]), sigma = np.array([[1.0, 0.5], [0.5, 4.25]]), nums = 5000)
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.scatter(np_gau[:,0], np_gau[:,1])
    plt.title("numpy gaussian two dim")
    plt.subplot(1, 2, 2)
    plt.scatter(my_gau[:,0], my_gau[:,1])
    plt.title("my gaussian two dim")
    plt.show()


def main():
    draw_one_dim_gau()
    draw_two_dim_gau()
    
main()
