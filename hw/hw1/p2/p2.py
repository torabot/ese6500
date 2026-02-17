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

HMM_init = HMM(z_hist, T_init, M_init, pi_init)


"""
Helper Functions
"""
# Generates LaTeX bmatrix from numpy array (from ChatGPT)
def to_bmatrix(A, fmt="{:g}"):
    """
    Convert a 1D or 2D numpy array into a LaTeX bmatrix string.

    Args:
        A: array-like (1D or 2D)
        fmt: format string or callable used to format each entry.
             Examples: "{:.3f}", "{:g}", or a function x -> "..."

    Returns:
        str: LaTeX code for \\begin{bmatrix} ... \\end{bmatrix}
    """
    A = np.asarray(A)

    if A.ndim == 0:
        A = A.reshape(1, 1)
    elif A.ndim == 1:
        A = A.reshape(1, -1)
    elif A.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {A.shape} (ndim={A.ndim}).")

    if callable(fmt):
        f = fmt
    else:
        f = lambda x: fmt.format(x)

    rows = []
    for row in A:
        rows.append(" & ".join(f(x) for x in row))

    body = " \\\\\n".join(rows)
    return "\\begin{bmatrix}\n" + body + "\n\\end{bmatrix}"


# Generates latex table from numpy array (from ChatGPT)
def numpy_to_latex_table(arr, row_label="t", col_prefix="x"):
    rows, cols = arr.shape
    
    header = " & ".join([row_label] + [f"{col_prefix}_{j+1}" for j in range(cols)]) + r" \\ \hline"
    
    body = []
    for i in range(rows):
        row = " & ".join([str(i)] + [f"{arr[i,j]:.3e}" for j in range(cols)])
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
Part a
"""
alpha = HMM_init.forward()
beta = HMM_init.backward()
gamma = HMM_init.gamma_comp(alpha, beta)

gamma_sums = gamma.sum(axis=1) # check if sum_{x} {gamma_k(x)} = 1

print("\n====== Part a =====\n")
print("\nGamma history:")
print(numpy_to_latex_table(gamma))

print("\nAlpha history:")
print(numpy_to_latex_table(alpha))

print("\nBeta history:")
print(numpy_to_latex_table(beta))

# print(f"\n Gammas for all timesteps: {gamma_sums}")

likely_states = np.where(gamma[:, 0] > gamma[:, 1], "LA", "NY")
ans_states = ["LA", "LA", "LA", "LA", "NY", "LA", "NY", "NY", "NY", "LA", "NY", "NY", "NY", "NY", "NY", "LA", "LA", "LA", "LA", "NY"]
print(f"\nMost likely states: {likely_states}")

# if np.array_equal(likely_states, ans_states):
#     print("Correct!")
# else:
#     print("Incorrect!")

"""
Part b
"""
# See written submission.


"""
Part c
"""
xi = HMM_init.xi_comp(alpha, beta, gamma)
T_prime, M_prime, pi_prime = HMM_init.update(alpha, beta, gamma, xi)

print(f"Size of xi is {xi.shape}")

print("\n====== Part c =====\n")

print("$\lambda$:")
print(f"""\\begin{{align*}}
\\pi &= {to_bmatrix(pi_init)} \\\\
T &= {to_bmatrix(T_init)} \\\\
M &= {to_bmatrix(M_init)}
\\end{{align*}}""")

print("\n$\lambda'$:")
print(f"""\\begin{{align*}}
\\pi' &= {to_bmatrix(pi_prime)} \\\\
T' &= {to_bmatrix(T_prime)} \\\\
M' &= {to_bmatrix(M_prime)}
\\end{{align*}}""")


"""
Part d
"""
print("\n====== Part d =====\n")

p_y_given_lambda = np.sum(alpha[-1, :])

HMM_prime = HMM(z_hist, T_prime, M_prime, pi_prime)
alpha_prime = HMM_prime.forward()
p_y_given_lambda_prime = np.sum(alpha_prime[-1, :])

print(f"P(Y_{{1:t}} | \\lambda) = {p_y_given_lambda}")
print(f"P(Y_{{1:t}} | \\lambda') = {p_y_given_lambda_prime}")