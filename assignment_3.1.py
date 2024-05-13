import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # distribution and sampling definition
    N = np.array([10,100,1000,10000])
    mu = np.array([-0.4,1.1])
    cov_matrix = np.array([[1,0.4],[0.4,1]])
    mu_est = np.zeros(np.shape(mu))
    cov_est = np.zeros(np.shape(cov_matrix))
    rmse_res = np.zeros(np.shape(N))
    abs_error_diag = np.zeros(np.shape(N))
    abs_error_off_diag = np.zeros(np.shape(N))
    
    for n in N:
        samples = np.random.multivariate_normal(mu,cov_matrix,n)
        # Monte carlo mean estimator
        mu_est[0] = np.sum(samples[:,0])/n
        mu_est[1] = np.sum(samples[:,1])/n
        # Monte carlo covariance estimator
        cov_est[0, 0] = np.sum((samples[:,0]-mu_est[0])*(samples[:,0]-mu_est[0]))/(n-1)
        cov_est[1, 1] = np.sum((samples[:,1]-mu_est[1])*(samples[:,1]-mu_est[1]))/(n-1)
        cov_est[0, 1] = np.sum((samples[:,0]-mu_est[0])*(samples[:,1]-mu_est[1]))/(n-1)
        cov_est[1, 0] = cov_est[0, 1]
        print(f'For N = {n}, mu_est = {mu_est}, cov_est = {cov_est}')
        rmse = np.sqrt((1/(n-1)*np.sum(np.power(samples[:,0]-mu_est[0],2)))/n) # for the first dimension only
        rmse_res[np.where(N==n)] = rmse
        abs_error_diag[np.where(N==n)] = np.abs(cov_est[0,0]-cov_matrix[0,0])
        abs_error_off_diag[np.where(N==n)] = np.abs(cov_est[0,1]-cov_matrix[0,1])

    fig = plt.figure()
    plt.plot(N,rmse_res, label='RMSE')
    plt.plot(N,abs_error_diag, label='Abs error diagonal')
    plt.plot(N,abs_error_off_diag, label='Abs error off-diagonal')
    plt.xscale('log')
    plt.title('RMSE of the mean estimator vs N w/ abs errors for covariance')
    plt.xlabel('N')
    plt.ylabel('RMSE & Abs errors')
    plt.grid()
    plt.legend()
    fig.savefig('assignment_3.1_rmse.png', dpi=fig.dpi)
