import numpy as np
import matplotlib.pyplot as plt
from als.myALS import myALS
from als.gValue import gValue

def driver_ALS_test():

    # Parameters
    lambda_val = 0.01
    nits = 500
    f = 2

    # Toy ratings matrix
    R = np.array([
        [5, 0, 0],
        [0, 4, 0],
        [4, 5, 1],
        [0, 0, 2],
        [0, 2, 0]
    ])

    # Highly accurate "true" prediction (from MATLAB)
    Rpred_accurate = np.array([
        [4.9861, 6.2009, 1.2467],
        [3.2106, 3.9928, 0.80279],
        [3.9993, 4.9737, 1],
        [1.2706, 1.5802, 1.9859],
        [1.6053, 1.9964, 0.40139]
    ])

    # Initialize
    U = np.array([[1,2,3,4,5],
                  [1,2,3,4,5]], dtype=float)
    M = np.ones((f, 3))

    err = []
    g_vals = []

    for it in range(nits):
        U, M = myALS(R, U, M, lambda_val)
        R_pred = U.T @ M
        err.append(np.linalg.norm(R_pred - Rpred_accurate))
        g_vals.append(gValue(R, U, M, lambda_val))

    # Error plot
    plt.figure()
    plt.plot(np.log10(err))
    plt.xlabel("Iterations")
    plt.title("log10 Error")
    plt.show()

    # g function plot
    plt.figure()
    plt.plot(g_vals)
    plt.xlabel("Iterations")
    plt.title("Value of g(U, M)")
    plt.show()

    # Final predicted ratings
    R_pred = U.T @ M
    print("Final predicted ratings:\n", R_pred)

if __name__ == "__main__":
    driver_ALS_test()