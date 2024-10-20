import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import scipy.special

def gaussian(x, mu, sigma):
    """1次元ガウス分布の確率密度関数"""
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2))

def log_gaussian(x, mu, sigma):
    """1次元ガウス分布の正規確率密度関数の対数"""
    return -1/2 * np.log(2*np.pi) -np.log(sigma)  -((x - mu)**2 / (2 * sigma**2))
 
def invgamma(x, alpha, beta):
    """逆ガンマ分布の確率密度関数"""
    return beta**alpha / scipy.special.gamma(alpha) * x**(-alpha - 1) * np.exp(-beta / x)

def gibbs_sampling(X, K=2, iterations=1000):
    """ギブスサンプリングの実装"""
    N = len(X)

    # 事前分布のパラメータ
    mu0 = 0  # 平均の事前分布の平均
    nu0 = 1  # 平均の事前分布の精度
    alpha0 = 2  # 分散の事前分布の形状パラメータ
    beta0 = 1  # 分散の事前分布の尺度パラメータ
    gamma = np.ones(K)  # 混合比率の事前分布の濃度パラメータ

    # パラメータの初期化
    mu = np.random.normal(mu0, 1 / np.sqrt(nu0), K)
    sigma2 = 1 / np.random.gamma(alpha0, 1 / beta0, K)
    pi = np.random.dirichlet(gamma)
    z = np.random.choice(K, N)      

    # ギブスサンプリングの繰り返し
    for _ in range(iterations):
        #zの更新
        for n in range(N):
            log_p = np.log(pi) + log_gaussian(X[n], mu, np.sqrt(sigma2))
            log_p_shifted = log_p - scipy.special.logsumexp(log_p)
            p = np.exp(log_p_shifted)
            z[n] = np.random.choice(K, p=p)

        # muの更新
        for k in range(K):
            Nk = np.sum(z == k)
            xbar = np.sum(X[z == k]) / Nk if Nk > 0 else 0
            mun = (nu0 * mu0 + Nk * xbar) / (nu0 + Nk)
            nun = nu0 + Nk
            mu[k] = np.random.normal(mun, 1 / np.sqrt(nun))

        # sigma2の更新
        for k in range(K):
            Nk = np.sum(z == k)
            ss = np.sum((X[z == k] - mu[k])**2) if Nk > 0 else 0
            alphan = alpha0 + Nk / 2
            betan = beta0 + ss / 2
            sigma2[k] = 1 / np.random.gamma(alphan, 1 / betan)

        # piの更新
        pi = np.random.dirichlet(gamma + np.bincount(z, minlength=K))

    return mu, np.sqrt(sigma2), pi, z

# サンプルデータの生成
np.random.seed(0)
data = np.concatenate([np.random.normal(-20, 1, 1000), np.random.normal(2, 1, 1000), np.random.normal(10, 1, 1000)])
#data = np.concatenate([np.random.normal(-2, 1, 1000), np.random.normal(2, 1, 1000)])

# ギブスサンプリングの実行
mu, sigma, pi, z = gibbs_sampling(X=data,K=3)

print("推定されたmu:", mu)
print("推定されたsigma:", sigma)
print("推定されたpi:", pi)

# グラフ表示
# 混合正規分布を計算するためのxの範囲
x = np.linspace(-50, 50, 1000)  
# 混合正規分布の確率密度関数
pdf = 0
for k in range(3):
    pdf += pi[k] * gaussian(x, mu[k], np.sqrt(sigma[k]*sigma[k]))

# プロット
fig = plt.figure(figsize=(10, 6))
plt.plot(x, pdf, label=f'1次元混合正規分布{_}番目')

# ヒストグラムを経験分布としてプロット
plt.hist(data, bins=100, density=True, alpha=0.5, label='データのヒストグラム')

plt.xlabel('x')
plt.ylabel('確率密度')
plt.title('1次元混合正規分布とデータのヒストグラム')
plt.legend()
plt.grid(alpha=0.4)
plt.show()
plt.close(fig)