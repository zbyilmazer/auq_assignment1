import numpy as np
import chaospy as cp
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.stats import beta

if __name__ == '__main__':
    # Reference result of the integral of exp(x) from 0 to 1
    ref_result = integrate.quad(lambda x: np.exp(x),0,1)[0]

    N = np.array([10,100,1000,10000])
    mc_rel_arr = np.zeros(len(N))
    cv_rel_arr_x = np.zeros(len(N))
    cv_rel_arr_x1 = np.zeros(len(N))
    beta_rel_arr_a5 = np.zeros(len(N))
    beta_rel_arr_a05 = np.zeros(len(N))
    for n in N:
        # Monte Carlo approximation
        uniform = cp.Uniform(0,1)
        samples = uniform.sample(size=n)
        mc_approx = np.sum(np.exp(samples))/n
        rel_err = np.abs(1-mc_approx/ref_result)
        mc_rel_arr[np.where(N==n)] = rel_err

        # Control variates approximation with g(x) = x
        samples_cv = uniform.sample(size=n)
        g_integral = integrate.quad(lambda x: x, 0, 1)[0]
        g_mean = np.mean(samples_cv)
        # choose the optimal c value
        c = -np.cov(samples, samples_cv)[0,1]/np.var(samples_cv)
        cv_approx = mc_approx + c*(g_integral-g_mean)
        rel_err = np.abs(1-cv_approx/ref_result)
        cv_rel_arr_x[np.where(N==n)] = rel_err

        # Control variates approximation with g(x) = x+1
        samples_cv = uniform.sample(size=n)
        g_integral = integrate.quad(lambda x: x+1, 0, 1)[0]
        g_mean = np.mean(samples_cv+1)
        # choose the optimal c value
        c = -np.cov(samples, samples_cv)[0,1]/np.var(samples_cv)
        cv_approx = mc_approx + c*(g_integral-g_mean)
        rel_err = np.abs(1-cv_approx/ref_result)
        cv_rel_arr_x1[np.where(N==n)] = rel_err

        # Importance sampling with a beta distribution with a = 5 & b = 1
        beta_dist = beta(5,1)
        samples = beta_dist.rvs(size=n)
        is_approx = np.sum(np.exp(samples)/beta_dist.pdf(samples))/n
        rel_err = np.abs(1-is_approx/ref_result)
        beta_rel_arr_a5[np.where(N==n)] = rel_err

        # Importance sampling with a beta distribution with a = 0.5 & b = 0.5
        beta_dist = beta(0.5,0.5)
        samples = beta_dist.rvs(size=n)
        is_approx = np.sum(np.exp(samples)/beta_dist.pdf(samples))/n
        rel_err = np.abs(1-is_approx/ref_result)
        beta_rel_arr_a05[np.where(N==n)] = rel_err

    fig = plt.figure(figsize=(10,6), dpi=300)
    plt.loglog(N,mc_rel_arr, label='MC')
    plt.loglog(N,cv_rel_arr_x, label='CV x')
    plt.loglog(N,cv_rel_arr_x1, label='CV x+1')
    plt.loglog(N,beta_rel_arr_a5, label='IS beta(5,1)')
    plt.loglog(N,beta_rel_arr_a05, label='IS beta(0.5,0.5)')
    plt.title('Relative error of the integral approximations vs N')
    plt.xlabel('N')
    plt.ylabel('Relative error')
    plt.grid()
    plt.legend()
    fig.savefig('assignment_4.1.png')







         
