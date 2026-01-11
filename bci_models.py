# bci_models.py
import numpy as np
from scipy import signal

class HeadModel:
    """
    Simulates the spatial properties of an 8-channel EEG headset based on the 10-20 system.
    This class is the core of the "Digital Twin" concept, creating realistic spatial
    correlation and topographical features for all generated signals.
    """
    def __init__(self, sampling_rate=250):
        self.sampling_rate = sampling_rate
        # Standard 8-channel OpenBCI layout (approximate 3D cartesian coordinates)
        # Channels:    1      2      3      4      5      6      7      8
        self.channel_names = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']
        self.electrode_positions = np.array([
            [-0.3, 0.9, 0.3], [0.3, 0.9, 0.3],  # 1, 2: Frontal-Polar (near eyes)
            [-0.8, 0.0, 0.6], [0.8, 0.0, 0.6],  # 3, 4: Central (motor cortex)
            [-0.7, -0.7, 0.2], [0.7, -0.7, 0.2], # 5, 6: Parietal (sensory)
            [-0.4, -0.9, 0.2], [0.4, -0.9, 0.2]  # 7, 8: Occipital (visual cortex)
        ])
        self.num_channels = len(self.channel_names)

        # --- Generate High-Fidelity Data-Driven Templates ---
        self.templates = self._generate_templates()

    def _generate_templates(self):
        """Creates realistic, procedural models of common artifacts."""
        templates = {}
        sr = self.sampling_rate

        # Blink Template: Models the sharp positive deflection and slower recovery.
        blink_duration = int(0.4 * sr)
        t = np.linspace(0, 1, blink_duration)
        rise = signal.windows.hann(int(len(t) * 0.4))[:int(len(t) * 0.2)]
        fall = signal.windows.hann(int(len(t) * 1.6))[:int(len(t) * 0.8)]
        blink = np.concatenate([rise, fall])
        templates['blink'] = (blink / np.max(blink)) * 120  # Scale to ~120 uV

        # Jaw Clench Template: Models a realistic EMG burst (bandpass-filtered noise).
        jaw_duration = int(1.2 * sr)
        jaw_noise = np.random.normal(0, 1, jaw_duration)
        sos = signal.butter(8, [20, 60], 'bandpass', fs=sr, output='sos')
        templates['jaw_clench'] = signal.sosfilt(sos, jaw_noise) * 50 # Scale to ~50 uV
        
        return templates

    def get_spatial_falloff(self, source_pos):
        """
        Calculates how much a signal from a source position decays at each electrode.
        This uses the inverse square law, creating a realistic topographical map.
        """
        distances = np.linalg.norm(self.electrode_positions - source_pos, axis=1)
        return 1 / (1 + (distances * 1.5)**2) # Damped inverse square law

    def get_source_locations(self):
        """Returns standard 3D locations for common signal sources."""
        return {
            'frontal_eyes':     np.array([0, 1.0, 0.25]),     # Source of blinks is near the eyes
            'motor_left':       np.array([-0.85, 0.0, 0.8]),  # Area near C3
            'motor_right':      np.array([0.85, 0.0, 0.8]),   # Area near C4
            'occipital_visual': np.array([0, -1.0, 0.3]),     # Source of alpha rhythms
            'global_tension':   np.array([0, 0, 1.0])         # Source of jaw clench is widespread
        }