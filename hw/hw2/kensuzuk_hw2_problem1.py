import numpy as np
from datetime import datetime

def ekf(a_bar_init, x_bar_init, Sigma_init, y, R = 1.0, Q = 0.5):

    mu_k_given_k = np.array([x_bar_init, a_bar_init])
    Sigma_k_given_k = Sigma_init

    x_bar_k_given_k = mu_k_given_k[0]
    a_bar_k_given_k = mu_k_given_k[1]

    N = y.shape[0]
    a_means = np.zeros(N)
    a_vars = np.zeros(N)

    for k in range(N):
        A = np.array([[a_bar_k_given_k, x_bar_k_given_k],
                     [0, 1.0]])
        x_bar_kp1_given_k = a_bar_k_given_k * x_bar_k_given_k
        a_bar_kp1_given_k = a_bar_k_given_k
        mu_kp1_given_k = np.array([x_bar_kp1_given_k, a_bar_kp1_given_k])

        Sigma_kp1_given_k = A @ Sigma_k_given_k @ A.T + np.array([[R, 0.0], [0.0, 0.0]])

        C = np.array([x_bar_kp1_given_k / (np.sqrt(x_bar_kp1_given_k**2.0 + 1.0))
                      , 0.0])
        
        g_of_mu_kp1_given_k = np.sqrt(x_bar_kp1_given_k**2.0 + 1.0)

        K = Sigma_kp1_given_k @ C.T / (C @ Sigma_kp1_given_k @ C.T + Q)

        mu_kp1_given_kp1 = mu_kp1_given_k + K * (y[k]- g_of_mu_kp1_given_k)
        Sigma_kp1_given_kp1 = (np.eye(2) - np.outer(K, C)) @ Sigma_kp1_given_k

        mu_k_given_k = mu_kp1_given_kp1
        Sigma_k_given_k = Sigma_kp1_given_kp1

        x_bar_k_given_k = mu_k_given_k[0]
        a_bar_k_given_k = mu_k_given_k[1]

        a_means[k] = a_bar_k_given_k
        a_vars[k] = Sigma_k_given_k[1, 1]

    return a_means, a_vars # return a_k_given_k 


def simulate_system(a, N, x0=0.0, seed=None, clip_sqrt=True, clip_min=1e-12):
    """
    Simulate:
        x_{k+1} = a x_k + eps_k,   eps_k ~ N(0, 1)
        y_k     = sqrt(x_k^2 + 1 + nu_k), nu_k ~ N(0, 0.5)

    Returns:
        x: shape (N+1,) states from x_0 ... x_N
        y: shape (N,)   measurements y_0 ... y_{N-1}
        eps: shape (N,) process noise samples
        nu: shape (N,)  measurement noise samples
    """
    rng = np.random.default_rng(seed)

    # Noise samples
    eps = rng.normal(loc=0.0, scale=np.sqrt(1.0), size=N)                  # var = 1
    nu  = rng.normal(loc=0.0, scale=np.sqrt(0.5), size=N)         # var = 0.5

    x = np.empty(N + 1)
    y = np.empty(N)

    x[0] = x0
    for k in range(N):
        # state update to x_{k+1}
        x[k + 1] = a * x[k] + eps[k]

        # measurement update
        square_root = x[k]**2 + 1.0
        if clip_sqrt:
            square_root = max(square_root, clip_min)  # avoid negative under sqrt
        y[k] = np.sqrt(square_root) + nu[k]

    return x, y, eps, nu


if __name__ == "__main__":


# Part a)
    print("===========================================================")
    print("Part a)")
    seed = 42069
    rng = np.random.default_rng(seed)
    x0 = rng.normal(loc=1.0, scale=np.sqrt(2.0), size=1)
    a = -1.0
    N = 100 # number of observations to run
    x, y, eps, nu = simulate_system(a=a, N=N, x0=0.0, seed=seed)
    export_dict = {"x": x, "y": y, "eps": eps, "nu": nu}
    file_name = f"p1_data_{datetime.now().strftime('%Y%m%d')}.npz"
    export_dir = "./p1_data/" + file_name
    np.savez(export_dir, **export_dict)

# Part b)
    print("===========================================================")
    print("Part b)")
    a_bar_init = 0.0

