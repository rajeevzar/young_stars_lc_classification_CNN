import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path




# Directories containing time series data
BEAT_DIR = "beat_periodic_lc/augmented_data/"
MULTI_DIR = "multi_lc/augmented_data/"
SINGLE_DIR = "single_lc/augmented_data/"
TRANSITS_DIR = "transits/augmented_data/"

# Directories to save images
BEAT_IMG_DIR = "beat_periodic_images/"
MULTI_IMG_DIR = "multi_images/"
SINGLE_IMG_DIR = "single_images/"
TRANSITS_IMG_DIR = "transits_images/"


# Create image directories if they don't exist
os.makedirs(BEAT_IMG_DIR, exist_ok=True)
os.makedirs(MULTI_IMG_DIR, exist_ok=True)
os.makedirs(SINGLE_IMG_DIR, exist_ok=True)
os.makedirs(TRANSITS_IMG_DIR, exist_ok=True)

# Parameters
NUM_SEGMENTS = 10  # Split each time series into 10 segments
IMG_SIZE = (128, 128)  # Image resolution

# Function to process and save images
def process_time_series(file_path, output_dir, class_name):
    # Load time series data
    data = np.loadtxt(file_path)
    time, flux = data[:, 0], data[:, 1]
    
    # Normalize time values to range [0,1]
    time = (time - np.min(time)) / (np.max(time) - np.min(time))
    
    # Split into NUM_SEGMENTS parts
    segment_length = len(time) // NUM_SEGMENTS
    
    for i in range(NUM_SEGMENTS):
        start_idx = i * segment_length
        end_idx = (i + 1) * segment_length if i < NUM_SEGMENTS - 1 else len(time)
        
        seg_time = time[start_idx:end_idx]
        seg_flux = flux[start_idx:end_idx]
        
        # Plot settings for CNN training
        plt.figure(figsize=(1.28, 1.28), dpi=100)
        plt.plot(seg_time, seg_flux, color='black', linewidth=2)
        plt.axis('off')  # Remove axis for better CNN feature extraction
        plt.margins(0)   # Reduce white space around the plot
        
        # Save image
        file_name = f"{class_name}_{Path(file_path).stem}_seg{i}.png"
        save_path = os.path.join(output_dir, file_name)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()

# Function to process all categories
def process_category(category_dir, category_img_dir, category_name):
    total_files = len(os.listdir(category_dir))
    print(f"Processing {total_files} {category_name} light curves...")
    for file in os.listdir(category_dir):
        if file.endswith(".dat"):
            process_time_series(os.path.join(category_dir, file), category_img_dir, category_name)

# Process each category
process_category(BEAT_DIR, BEAT_IMG_DIR, "beat_periodic")
process_category(MULTI_DIR, MULTI_IMG_DIR, "multi")
process_category(SINGLE_DIR, SINGLE_IMG_DIR, "single")
process_category(TRANSITS_DIR, TRANSITS_IMG_DIR, "transits")

print("Image generation complete!")
