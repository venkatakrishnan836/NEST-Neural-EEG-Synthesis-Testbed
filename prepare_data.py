#
# --- FILE: prepare_data.py (v8 - Final MNE API Fix) ---
#
import numpy as np
import mne
from scipy.io import loadmat
import os

# --- Configuration ---
DATASET_PATH = './raw_als_data/'
TARGET_CHANNELS_INDICES = [0, 1, 2, 3, 4, 5, 6, 7]
SAMPLING_RATE = 256
TARGET_SR = 250
WINDOW_SIZE = TARGET_SR * 1
WINDOW_STEP = WINDOW_SIZE // 4

def preprocess_eeg_data(data):
    """Applies filtering, resampling, and normalization."""
    data_float64 = data.astype(np.float64)
    
    filtered_data = mne.filter.filter_data(data_float64, sfreq=SAMPLING_RATE, l_freq=1.0, h_freq=45.0, verbose=False)
    
    # ===================================================================
    # THE FINAL MNE API FIX:
    # Replace the outdated 'sfreq_new' with the modern 'up' and 'down' arguments.
    # This tells MNE the exact ratio for resampling.
    # ===================================================================
    resampled_data = mne.filter.resample(filtered_data, up=TARGET_SR, down=SAMPLING_RATE, npad="auto", verbose=False)
    # ===================================================================

    car_data = resampled_data - np.mean(resampled_data, axis=0, keepdims=True)
    
    min_val, max_val = np.min(car_data), np.max(car_data)
    if np.isclose(max_val, min_val): return np.zeros_like(car_data), min_val, max_val
    normalized_data = 2 * ((car_data - min_val) / (max_val - min_val)) - 1
    return normalized_data, min_val, max_val

def create_windows(data, window_size, step):
    """Creates overlapping windows."""
    windows = []
    num_samples = data.shape[1]
    for i in range(0, num_samples - window_size, step):
        windows.append(data[:, i:i + window_size])
    return np.array(windows)

# --- Main Script ---
print("Starting EEG data preparation (v8 - MNE Resample Fix)...")
all_windows = []
norm_params_set = False

for filename in os.listdir(DATASET_PATH):
    if filename.endswith('.mat'):
        print(f"Processing file: {filename}")
        mat = loadmat(os.path.join(DATASET_PATH, filename))
        
        ignore_keys = ['__header__', '__version__', '__globals__']
        data_key = [k for k in mat.keys() if k not in ignore_keys][0]
        data_struct = mat[data_key][0, 0]

        left_data = data_struct['L']
        right_data = data_struct['R']
        rest_data = data_struct['Re']

        combined_data_raw = np.vstack([left_data, right_data, rest_data])
        raw_eeg_data = combined_data_raw.T
        
        selected_data = raw_eeg_data[TARGET_CHANNELS_INDICES, :]

        normalized_data, min_val, max_val = preprocess_eeg_data(selected_data)
        
        if not norm_params_set and not np.isclose(max_val, min_val):
            np.save('als_eeg_norm_params.npy', np.array([min_val, max_val]))
            norm_params_set = True
            print(f"  --> Normalization parameters set from {filename} and saved.")

        windows = create_windows(normalized_data, WINDOW_SIZE, WINDOW_STEP)
        all_windows.append(windows)

if not all_windows:
    print("\nERROR: No data was processed.")
else:
    final_dataset = np.concatenate(all_windows, axis=0)
    final_dataset = np.transpose(final_dataset, (0, 2, 1))
    np.save('als_eeg_processed_dataset.npy', final_dataset)
    print(f"\nData preparation complete. Shape of final dataset: {final_dataset.shape}")
    print("Saved to 'als_eeg_processed_dataset.npy' and 'als_eeg_norm_params.npy'")