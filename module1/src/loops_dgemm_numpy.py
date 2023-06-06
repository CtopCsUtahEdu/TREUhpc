import time
import numpy as np

m = 1024
n = 1024
k = 1024
numIterations = 10

# Allocate memory for matrices
A = np.zeros((m, k))
B = np.zeros((k, n))
C = np.zeros((m, n))

# Initialize input matrices
for i in range(m):
    for j in range(k):
        A[i, j] = i + 1

for i in range(k):
    for j in range(n):
        B[i, j] = -(i + 1)

# Create variables for timing
totalTime = 0.0

for iter in range(numIterations):
    # Clear the result matrix
    C.fill(0.0)

    # Start the timer
    startTime = time.time()

    # Perform matrix multiplication using NumPy
    C = np.matmul(A, B)

    # Stop the timer
    endTime = time.time()

    # Calculate elapsed time
    elapsedTime = endTime - startTime
    totalTime += elapsedTime

# Calculate average elapsed time
averageTime = totalTime / numIterations

# Calculate number of floating-point operations (FLOPs)
flops = (2.0e-9 * m * n * k) / averageTime

# Print the average elapsed time and FLOPs
print(f"Matrix multiplication using NumPy: A({m}, {k}) * B({k}, {n}) = C({m}, {n})")
print(f"Average execution time: {averageTime:.5f} seconds")
print(f"Performance in GFLOPs: {flops:.5f} GFLOPs")
