import time

def matmul(A, B):
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])

    if cols_A != rows_B:
        raise ValueError("The number of columns in A must match the number of rows in B")

    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]

    return result

# Create two example matrices of size 1000x1000
A = [[i * 1000 + j + 1 for j in range(1000)] for i in range(1000)]
B = [[j * 1000 + i + 1 for j in range(1000)] for i in range(1000)]

start_time = time.time()
result = matmul(A, B)
end_time = time.time()
execution_time = end_time - start_time

print("Execution time: %.6f seconds" % execution_time)

