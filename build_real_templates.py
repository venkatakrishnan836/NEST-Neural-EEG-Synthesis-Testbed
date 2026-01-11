import mne
import numpy as np
import os

print("MNE-Python library imported successfully.")

# --- Configuration ---
TEMPLATE_DIR = "template_library"
SUBJECT_ID = 1
RUN_ID = 2 

if not os.path.exists(TEMPLATE_DIR):
    os.makedirs(TEMPLATE_DIR)

print(f"Attempting to download/load data for Subject {SUBJECT_ID}, Run {RUN_ID} from PhysioNet...")

try:
    # --- Step 1 & 2: Download and Load Data ---
    # This will use the cached version since you've already downloaded it.
    raw_filepath = mne.datasets.eegbci.load_data(SUBJECT_ID, RUN_ID)[0]
    raw = mne.io.read_raw_edf(raw_filepath, preload=True, exclude=['stim'])
    print("Data loaded successfully.")
    
    # We rename the channels to match MNE's standard naming convention
    # This is a common and necessary step for many datasets.
    mne.rename_channels(raw.info, {'Fc5.': 'FC5', 'Fc3.': 'FC3', 'Fc1.': 'FC1', 'Fcz.': 'FCz',
     'Fc2.': 'FC2', 'Fc4.': 'FC4', 'Fc6.': 'FC6', 'C5..': 'C5', 'C3..': 'C3',
     'C1..': 'C1', 'Cz..': 'Cz', 'C2..': 'C2', 'C4..': 'C4', 'C6..': 'C6',
     'Cp5.': 'CP5', 'Cp3.': 'CP3', 'Cp1.': 'CP1', 'Cpz.': 'CPz', 'Cp2.': 'CP2',
     'Cp4.': 'CP4', 'Cp6.': 'CP6', 'Fp1.': 'Fp1', 'Fpz.': 'Fpz', 'Fp2.': 'Fp2',
     'Af7.': 'AF7', 'Af3.': 'AF3', 'Afz.': 'AFz', 'Af4.': 'AF4', 'Af8.': 'AF8',
     'F7..': 'F7', 'F5..': 'F5', 'F3..': 'F3', 'F1..': 'F1', 'Fz..': 'Fz',
     'F2..': 'F2', 'F4..': 'F4', 'F6..': 'F6', 'F8..': 'F8', 'Ft7.': 'FT7',
     'Ft8.': 'FT8', 'T7..': 'T7', 'T8..': 'T8', 'T9..': 'T9', 'T10.': 'T10',
     'Tp7.': 'TP7', 'Tp8.': 'TP8', 'P7..': 'P7', 'P5..': 'P5', 'P3..': 'P3',
     'P1..': 'P1', 'Pz..': 'Pz', 'P2..': 'P2', 'P4..': 'P4', 'P6..': 'P6',
     'P8..': 'P8', 'Po7.': 'PO7', 'Po3.': 'PO3', 'Poz.': 'POz', 'Po4.': 'PO4',
     'Po8.': 'PO8', 'O1..': 'O1', 'Oz..': 'Oz', 'O2..': 'O2', 'Iz..': 'Iz'})

    raw.set_montage('standard_1005', on_missing='warn')

    # --- Step 3: Find Eye Blinks (With Manual Override) ---
    print("Finding eye blink events in the real data (using Fp1 as reference)...")
    
    # THIS IS THE CRITICAL FIX: We explicitly tell MNE which channel to use.
    blink_events = mne.preprocessing.find_eog_events(raw, ch_name='Fp1')
    
    if len(blink_events) == 0:
        raise RuntimeError("No blinks found in this data run, even with manual channel selection.")
        
    print(f"Found {len(blink_events)} blinks. Creating an average 'Golden Signal' blink.")

    # --- Step 4 & 5: Create Epoch and Save Template (same as before) ---
    epochs = mne.Epochs(raw, blink_events, event_id=998, tmin=-0.25, tmax=0.25, baseline=None, preload=True)
    average_blink_epoch = epochs.average()

    fp1_channel_index = average_blink_epoch.ch_names.index('Fp1')
    real_blink_template = average_blink_epoch.data[fp1_channel_index]
    
    real_blink_template = real_blink_template * 1e6
    real_blink_template = real_blink_template - real_blink_template[0]

    template_path = os.path.join(TEMPLATE_DIR, "real_blink_template.npy")
    np.save(template_path, real_blink_template)
    
    print("\n----------------------------------------------------")
    print("SUCCESS!")
    print(f"A real, data-driven blink template has been saved to: {template_path}")
    print("You can now upgrade your mset_main.py to load this file.")
    print("----------------------------------------------------")

except Exception as e:
    print("\n----------------------------------------------------")
    print(f"An error occurred: {e}")
    print("----------------------------------------------------")