#-*- coding:utf-8 -*-
# ʹ��Rejection Sampling�ܾ�������������ϸ�˹�ֲ�
import numpy as np
import matplotlib.pyplot as plt

# numpy ��������ά�ȵĸ�˹�ֲ�
def numpy_gaussian(mu, sigma, nums):
    dims = mu.shape[0]
    # X = N0 * Sigma^0.5 + Mu
    return np.dot(np.random.randn(nums, dims), np.linalg.cholesky(sigma)) + mu

# ��˹�ֲ�����x���p(x|mu, sigma)
def single_gaussian_pdf(xs, mu, sigma):
    # @param xs����������ݵ㣬m * n����n=1άʱ��ҲҪ��(m, 1)��ʽ������
    # @param mu����ֵ��һά����
    # @param sigma�������ά����
    m = xs.shape[0]        # ���������
    pxs = np.zeros(m)      # ���ؽ����m������ֵ
    n = mu.shape[0]        # ��˹ά��
    
    frac = ((2 * np.pi) ** (n/2)) * np.sqrt(np.linalg.det(np.mat(sigma)))
    # ��������������
    for i in range(m):
        ux = xs[i, :] - mu
        px = np.exp(-0.5 * np.dot(np.dot(ux.reshape(1, n), np.mat(sigma).I.A), ux.reshape(n, 1)))/frac
        pxs[i] = px
    return pxs

# ��ϸ�˹�ֲ��ĸ����ܶȺ���
def mix_gaussian_pdf(xs, alphas, mus, sigmas):
    #@param xs     : (m, n)�����ݵ㣬numpy.array�� n = 1ʱҲ�Ƕ�ά���� 
    #@param alphas : ÿ����˹�ɷֵı��� length = K���б�
    #@param mus    : ÿ����˹�ľ�ֵ��length = K���б��б�Ԫ����һά����
    #@param sigmas : ÿ����˹�ɷֵķ��length = K���б��б�Ԫ����һ����ά����
    K = len(alphas)               # K����˹�ɷ�
    m = xs.shape[0]               # ���������
    
    pxs = np.zeros(m)
    for k in range(K):
        alpha = alphas[k]
        mu = mus[k]
        sigma = sigmas[k]
        pxs += alpha * single_gaussian_pdf(xs, mu, sigma)
    return pxs

# �ܾ�����
def rejection_sampling():
    # �Ե����ĸ�˹���в���
    xs = numpy_gaussian(mu = np.array([0.0]), sigma = np.array([[10.0]]), nums = 5000)
    xs = np.sort(xs, axis = 0)    # ����
    
    # ����x����single_gau��mix_gau��Ӧ�ĸ���
    p1 = 2.0 * single_gaussian_pdf(xs.reshape((-1, 1)), mu = np.array([0.0]), sigma = np.array([[10.0]]))
    
    alphas = [0.5, 0.27, 0.23]        # Ȩ��
    mus = [np.array([0.0]), np.array([-3.0]), np.array([3.0])]           # ��ֵ
    sigmas = [np.array([[1.0]]), np.array([[2.0]]), np.array([[0.5]])]   # ����
    p2 = mix_gaussian_pdf(xs.reshape((-1, 1)), alphas = alphas, mus = mus, sigmas = sigmas)
    # ���ھܾ������õ��Ļ�ϸ�˹�ֲ����
    mix_gau_nums = []
    for i, x in enumerate(xs):
        u = np.random.rand()
        if u <= p2[i] / p1[i]:
            mix_gau_nums.append(x)
    # ��ӡ����������
    print("Rejection sampling ratio = ", len(mix_gau_nums)/xs.shape[0])    # 0.4986
    print("Mean value = ", sum(mix_gau_nums)/len(mix_gau_nums))            # -0.08131822
    # ���ƽ��
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