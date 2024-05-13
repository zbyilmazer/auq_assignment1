import numpy as np
import chaospy as cp
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def model(init_cond, t, params):
    z0, z1 = init_cond
    c, k, f, w = params
    f = [z1, f * np.cos(w * t) - k * z0 - c * z1]
    return f

def discretize_oscillator_odeint(model, init_cond, t, params, atol, rtol):
    sol = odeint(model, init_cond, t, args=(params,), atol=atol, rtol=rtol)
    return sol

if __name__ == '__main__':
    c = 0.5
    k = 2.0
    f = 0.5
    w = 1.0
    y0 = 0.5
    y1 = 0.0
    t_max = 20.0
    dt = 0.01
    init_cond = y0, y1
    params_odeint = c, k, f, w
    atol = 1e-10
    rtol = 1e-10

    grid_size = int(t_max / dt) + 1
    t = np.linspace(0, t_max, grid_size)

    odeint_solution_deterministic = discretize_oscillator_odeint(
        model, init_cond, t, params_odeint, atol, rtol
    )
    print(f'For deterministic w = 1.0, y0 = {odeint_solution_deterministic[1000,0]}, y1 = {odeint_solution_deterministic[1000,1]} at t = 10')

    # Sample w from a uniform distribution
    N = np.array([10, 100, 1000, 10000])
    uniform = cp.Uniform(0.95, 1.05)
    trajectories = []
    rel_error = []
    ref_mean = np.array([-0.43893703, 0.04293818])
    for n in N:
        w_samples = uniform.sample(size=n)
        odeint_solutions = []
        for w in w_samples:
            params = c, k, f, w
            odeint_solution = discretize_oscillator_odeint(
                model, init_cond, t, params, atol, rtol
            )
            odeint_solutions.append(odeint_solution[1000,:])
            if n == 10 and len(trajectories) < 5:
                trajectories.append((w, odeint_solution))
        sol_mean_y0 = np.mean([sol[0] for sol in odeint_solutions])
        sol_mean_y1 = np.mean([sol[1] for sol in odeint_solutions])
        sol_var_y0 = np.var([sol[0] for sol in odeint_solutions])
        sol_var_y1 = np.var([sol[1] for sol in odeint_solutions])
        print(f'For N = {n}, mean y0 = {sol_mean_y0}, mean y1 = {sol_mean_y1}, var y0 = {sol_var_y0}, var y1 = {sol_var_y1} at t = 10')
        rel_error.append((np.abs(1-sol_mean_y0/ref_mean[0]), np.abs(1-sol_mean_y1/ref_mean[1])))

    fig = plt.figure(figsize=(10, 6), dpi=300)
    for w, trajectory in trajectories:
        plt.plot(t, trajectory[:,0], '--', label=f'w = {w}')
    plt.title('Trajectories for 5 samples')
    plt.xlabel('t')
    plt.ylabel('y0')
    plt.grid()
    plt.legend()
    fig.savefig('assignment_3.2_trajectories.png')

    fig = plt.figure() 
    plt.loglog(N, [err[0] for err in rel_error], label='Relative error y0')
    plt.loglog(N, [err[1] for err in rel_error], label='Relative error y1')
    plt.title('Relative error of the mean estimator vs N')
    plt.xlabel('N')
    plt.ylabel('Relative error')
    plt.grid()
    plt.legend()
    fig.savefig('assignment_3.2_rel_error.png', dpi=300)

