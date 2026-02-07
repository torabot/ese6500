import numpy as np


class HistogramFilter(object):
    """
    Class HistogramFilter implements the Bayes Filter on a discretized grid space.
    """

    def histogram_filter(self, cmap, belief, action, observation):
        '''
        Takes in a prior belief distribution, a colormap, action, and observation, and returns the posterior
        belief distribution according to the Bayes Filter.
        :param cmap: The binary NxM colormap known to the robot.
        :param belief: An NxM numpy ndarray representing the prior belief.
        :param action: The action as a numpy ndarray. [(1, 0), (-1, 0), (0, 1), (0, -1)]
        :param observation: The observation from the color sensor. [0 or 1].
        :return: The posterior distribution.
        '''

        ### Your Algorithm goes Below.
        N, M = belief.shape
        u_row, u_col = action

        belief_pred = np.zeros_like(belief)

        # Loop through all states
        for row in range(N):#-1, 0, -1):
            for col in range(M):#-1, 0, -1):

                # Prediction step
                row_next = row - u_col
                col_next = col + u_row

                hit_wall = (
                    (u_col == 1 and row == 0) or          # Action 'Right' (0, 1) -> move Up, check top wall
                    (u_col == -1 and row == N - 1) or     # Action 'Left' (0, -1) -> move Down, check bottom wall
                    (u_row == -1 and col == 0) or         # Action 'Up' (-1, 0) -> move Left, check left wall
                    (u_row == 1 and col == M - 1)         # Action 'Down' (1, 0) -> move Right, check right wall
                )
                belief_pred[row, col] += 0.1 * belief[row, col]

                if hit_wall:
                    belief_pred[row, col] += 0.9 * belief[row, col] 
                else:
                    # print(f"row_next, col_next: {row_next}, {col_next}")
                    belief_pred[row_next, col_next] += 0.9 * belief[row, col]

        # Measurement step
        bel_bar = np.where(cmap == observation, 0.9, 0.1)
        post_belief = belief_pred * bel_bar

        total_belief = np.sum(post_belief)

        if total_belief < 1e-10:
            total_belief = 1.0

        return post_belief / total_belief