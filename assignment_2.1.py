import numpy as np
import chaospy as cp
import matplotlib.pyplot as plt
import scipy.integrate as integrate

if __name__ == '__main__':
    N = np.array([10,100,1000,10000])
    rmse_array = np.zeros(len(N))
    exact_error = np.zeros(len(N))
    uniform = cp.Uniform(0,1)
    
    #Exact value of the integral
    exact_value = integrate.quad(lambda x: np.sin(x),0,1)[0] # second value is the error
    for n in N:
        # Monte Carlo approximation
        samples = uniform.sample(size=n)
        mc_approx = np.sum(np.sin(samples))/n
        rmse = np.sqrt((1/(n-1)*np.sum(np.power(np.sin(samples)-mc_approx,2)))/n)
        rmse_array[np.where(N==n)] = rmse
        exact_error[np.where(N==n)] = np.abs(exact_value-mc_approx)

    fig = plt.figure()
    plt.loglog(N,rmse_array)
    plt.title('RMSE of the mean estimator vs N')
    plt.xlabel('N')
    plt.ylabel('RMSE')
    plt.grid()
    fig.savefig('assignment_2.1_rmse.png', dpi=fig.dpi)

    fig = plt.figure()
    plt.plot(N,exact_error)
    plt.xscale('log')
    plt.title('Exact error vs N')
    plt.xlabel('N')
    plt.ylabel('Exact error')
    plt.grid()
    fig.savefig('assignment_2.1_exact_error.png', dpi=fig.dpi)