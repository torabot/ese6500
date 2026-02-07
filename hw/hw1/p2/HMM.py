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
    # forward algorithm
        alpha = np.zeros((self.horizon, self.num_states))
        alpha[0, :] = self.Initial_distribution * self.Emission[:, self.Observations[0]] # initialization

        for k in range(self.horizon - 1):
            alpha[k+1, :] = (alpha[k, :] @ self.Transition) * self.Emission[:, self.Observations[k+1]]

        return alpha

    def backward(self):

        beta = np.zeros((self.horizon, self.num_states))
        beta[self.horizon - 1, :] = np.ones(self.num_states) # initialization

        for k in range(self.horizon - 2, -1, -1):
            beta[k, :] = self.Transition @ (beta[k+1, :] * self.Emission[:, self.Observations[k+1]])

        return beta

    def gamma_comp(self, alpha, beta):

        gamma = alpha * beta

        eta = np.sum(alpha[-1, :])
        gamma /= eta

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

        new_init_state = ...
        T_prime = ...
        M_prime = ...

        return T_prime, M_prime, new_init_state

    def trajectory_probability(self, alpha, beta, T_prime, M_prime, new_init_state):

        P_original = ...
        P_prime = ...

        return P_original, P_prime
