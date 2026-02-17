import numpy as np

class HMM():

    def __init__(self, Observations, Transition, Emission, Initial_distribution):
        self.Observations = Observations
        self.Transition = Transition
        self.Emission = Emission
        self.Initial_distribution = Initial_distribution

        self.num_states = Transition.shape[0] # number of states
        self.num_obs = Emission.shape[1] # number of observations
        self.horizon = Observations.shape[0] # horizon


    def forward(self):
        alpha = np.zeros((self.horizon, self.num_states))
        alpha[0, :] = self.Initial_distribution * self.Emission[:, self.Observations[0]] # initialization

        for k in range(self.horizon - 1):
            alpha[k+1, :] = (alpha[k, :] @ self.Transition) * self.Emission[:, self.Observations[k+1]]

        return alpha


    def backward(self):
        beta = np.zeros((self.horizon, self.num_states))
        beta[self.horizon - 1, :] = np.ones(self.num_states)

        for k in range(self.horizon - 2, -1, -1):
            beta[k, :] = self.Transition @ (beta[k+1, :] * self.Emission[:, self.Observations[k+1]])

        return beta


    def gamma_comp(self, alpha, beta):
        gamma = alpha * beta
        eta = 1/np.sum(alpha[-1, :])
        gamma *= eta
        return gamma


    def xi_comp(self, alpha, beta, gamma):

        xi = np.zeros((self.horizon-1, self.num_states, self.num_states))

        for k in range(self.horizon - 1):
            for i in range(self.num_states):
                for j in range(self.num_states):
                    xi[k, i, j] = alpha[k, i] * self.Transition[i, j] * self.Emission[j, self.Observations[k+1]] * beta[k+1, j]

            eta = 1/np.sum(xi[k, :, :])
            if eta is not 0:
                xi[k, :, :] *= eta

        return xi

    def update(self, alpha, beta, gamma, xi):

        new_init_state = gamma[0, :]
        T_prime = np.sum(xi, axis=0) / np.sum(gamma[:-1, :], axis=0).reshape(-1, 1)

        M_prime = np.zeros((self.num_states, self.num_obs))
        for k in range(self.horizon):
            obs_k = self.Observations[k]
            M_prime[:, obs_k] += gamma[k, :]
        
        M_prime /= np.sum(gamma, axis=0).reshape(-1, 1)

        return T_prime, M_prime, new_init_state

    def trajectory_probability(self, alpha, beta, T_prime, M_prime, new_init_state):

        P_original = np.sum(alpha[-1, :])

        alpha_prime = np.zeros_like(alpha)
        alpha_prime[0, :] = new_init_state * M_prime[:, self.Observations[0]]
        for k in range(self.horizon - 1):
            alpha_prime[k+1, :] = (alpha_prime[k, :] @ T_prime) * M_prime[:, self.Observations[k+1]]
        P_prime = np.sum(alpha_prime[-1, :])

        return P_original, P_prime
