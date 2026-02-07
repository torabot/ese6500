import numpy as np
from HMM import HMM

"""
States (x):
- 0: LA
- 1: NY
)
Observations (z):
- 0: LA
- 1: NY
- 2: null
"""

T_init = np.array([
                  [0.5, 0.5],
                  [0.5, 0.5]
                  ])

M_init = np.array([
                  [0.4, 0.1, 0.5],
                  [0.1, 0.5, 0.4]
                  ])

z_hist = np.array([2, 0, 0, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 0, 0, 1])

pi_init = np.array([0.5, 0.5])

num_states = T_init.shape[0]
num_observations = M_init.shape[1]
seq_len = z_hist.shape[0]

HMM = HMM(z_hist, T_init, M_init, pi_init)


"""
Helper Functions
"""
# Generates latex table from numpy array (from ChatGPT)
def numpy_to_latex_table(arr, row_label="t", col_prefix="x"):
    rows, cols = arr.shape
    
    header = " & ".join([row_label] + [f"{col_prefix}_{j+1}" for j in range(cols)]) + r" \\ \hline"
    
    body = []
    for i in range(rows):
        row = " & ".join([str(i)] + [f"{arr[i,j]:.3f}" for j in range(cols)])
        body.append(row + r" \\")
    
    col_format = "c|" + "c"*cols
    
    table = [
        r"\begin{tabular}{" + col_format + "}",
        r"\hline",
        header,
        *body,
        r"\hline",
        r"\end{tabular}"
    ]
    
    return "\n".join(table)



"""
Part 1
"""
alpha = HMM.forward()
beta = HMM.backward()
gamma = HMM.gamma_comp(alpha, beta)

gamma_sums = gamma.sum(axis=1) # check if sum_{x} {gamma_k(x)} = 1

print("\n====== Part 1 =====\n")
print("\nGamma history:")
print(numpy_to_latex_table(gamma))

print("\nAlpha history:")
print(numpy_to_latex_table(alpha))

print("\nBeta history:")
print(numpy_to_latex_table(beta))

print(f"\n Gammas for all timesteps: {gamma_sums}")


"""
Part 2
"""
