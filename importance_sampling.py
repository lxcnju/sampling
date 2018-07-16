#-*- coding:utf-8 -*-
# ʹ��Importence Sampling����Ȩ�صز����������ֵ
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

# Ѱ��һ����˹�ֲ�������ȫ�ڻ�ϸ�˹�ֲ�����
def draw_single_mix_gaussian():
    x = np.linspace(-5, 5, 3000)
    alphas = [0.5, 0.27, 0.23]        # Ȩ��
    mus = [np.array([0.0]), np.array([-3.0]), np.array([3.0])]           # ��ֵ
    sigmas = [np.array([[1.0]]), np.array([[2.0]]), np.array([[0.5]])]   # ����
    
    mix_y = mix_gaussian_pdf(x.reshape((-1, 1)), alphas = alphas, mus = mus, sigmas = sigmas)
    
    single_y = 2.0 * single_gaussian_pdf(x.reshape((-1, 1)), mu = np.array([0.0]), sigma = np.array([[10.0]]))
    
    plt.figure()
    plt.plot(x, mix_y, 'b', x, single_y, 'r')
    plt.show()

# ��Ҫ�Բ���
def importance_sampling():
    # �Ե����ĸ�˹���в���
    xs = numpy_gaussian(mu = np.array([0.0]), sigma = np.array([[10.0]]), nums = 5000)
    xs = np.sort(xs, axis = 0)    # ����
    
    # ����x����single_gau��mix_gau��Ӧ�ĸ���
    p1 = 2.0 * single_gaussian_pdf(xs.reshape((-1, 1)), mu = np.array([0.0]), sigma = np.array([[10.0]]))
    
    alphas = [0.5, 0.27, 0.23]        # Ȩ��
    mus = [np.array([0.0]), np.array([-3.0]), np.array([3.0])]           # ��ֵ
    sigmas = [np.array([[1.0]]), np.array([[2.0]]), np.array([[0.5]])]   # ����
    p2 = mix_gaussian_pdf(xs.reshape((-1, 1)), alphas = alphas, mus = mus, sigmas = sigmas)
    
    # Ȩ��
    xs_weights = p2 / p1
    
    # �����ֵ
    mean_value = np.dot(xs.reshape(-1), xs_weights) / xs.shape[0]
    print("Mean value = ", mean_value)  # -0.0524097355004
    
def main():
    draw_single_mix_gaussian()
    importance_sampling()

main()