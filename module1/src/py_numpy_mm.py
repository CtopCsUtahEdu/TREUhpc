import numpy as np
import time

# Create two random matrices
A = np.random.rand(1000, 1000)
B = np.random.rand(1000, 1000)

# Start the timer
start_time = time.time()

# Perform matrix multiplication
C = np.matmul(A, B)

# Calculate the elapsed time
elapsed_time = time.time() - start_time

# Print the elapsed time
print("Elapsed time: {:.2f} seconds".format(elapsed_time))

