import matplotlib.pyplot as plt
import os
import numpy as np

# Directories for original and augmented data
data_dir = "./"
augmented_data_dir = "./augmented_data"

# Loop through all .dat files in the data directory
for file_name in os.listdir(data_dir):
    if file_name.endswith(".dat"):
        # Load original time series
        original_file_path = os.path.join(data_dir, file_name)
        original_series = np.loadtxt(original_file_path)  # Assuming 2D data: [time, values]

        # Load augmented time series
        augmented_file_path = os.path.join(augmented_data_dir, f"augmented_{file_name}")
        if os.path.exists(augmented_file_path):
            augmented_series = np.loadtxt(augmented_file_path)

            # Plot original and augmented series
            plt.figure(figsize=(10, 6))
            plt.plot(original_series[:, 0], original_series[:, 1], label="Original Series")
            plt.plot(augmented_series[:, 0], augmented_series[:, 1], label="Augmented Series")
            plt.legend()
            plt.xlabel("Time")
            plt.ylabel("Values")
            plt.title(f"Original vs Augmented: {file_name}")
            plt.show()
        else:
            print(f"Augmented file not found for: {file_name}")
