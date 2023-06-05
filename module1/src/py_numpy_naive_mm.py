import numpy as np
import time

def matmul(A, B):
    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape

    if cols_A != rows_B:
        raise ValueError("The number of columns in A must match the number of rows in B")

    result = np.zeros((rows_A, cols_B))

    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]

    return result

# Create two example matrices of size 1000x1000 using NumPy
A = np.arange(1, 1000001).reshape(1000, 1000)
B = np.arange(1000001, 2000001).reshape(1000, 1000)

# Convert the matrices to NumPy arrays
A = np.array(A)
B = np.array(B)

start_time = time.time()
result = matmul(A, B)
end_time = time.time()
execution_time = end_time - start_time

print("Execution time: %.6f seconds" % execution_time)
