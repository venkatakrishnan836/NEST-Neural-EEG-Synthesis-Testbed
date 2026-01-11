# mset_main.py (v7.0 - The Complete Mind OS Control Panel)
import sys, time as systime, numpy as np, random, os, struct
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QSlider, QLabel, QPushButton, QGridLayout, QCheckBox, QFrame)
from PyQt6.QtCore import QTimer, Qt
import pyqtgraph as pg
from scipy import signal
from pylsl import StreamInfo, StreamOutlet
from bci_models import HeadModel

# --- Import and check for ML libraries ---
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    
# --- Constants ---
HEADER_BYTE = 0xA0
FOOTER_BYTE = 0xC0
SCALE_FACTOR_uV_PER_COUNT = 0.02235
LATENT_DIM = 128

# =============================================================================
#  Signal Generation v7.0 - With Attention Control
# =============================================================================
class SignalGenerator:
    def __init__(self, sampling_rate=250):
        # ... (initialization is mostly the same) ...
        self.sampling_rate = sampling_rate
        self.head_model = HeadModel(sampling_rate)
        self.num_channels = self.head_model.num_channels
        self.time = 0
        self.components_enabled = {'rhythms': True, 'noise': True, 'environment': True}
        
        # --- [MODIFIED] Add 'attention' to the state dictionary ---
        self.state = {'load': 0.0, 'drowsy': 0.0, 'tension': 0.0, 'attention': 1.0} # Attention defaults to 100%

        self.use_gan = False
        if TENSORFLOW_AVAILABLE:
            print("TensorFlow found. Attempting to load GAN generator...")
            try:
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
                self.gan_generator = tf.keras.models.load_model('gan_models/generator.h5', compile=False)
                self.norm_params = np.load('als_eeg_norm_params.npy')
                self.generated_buffer = np.array([])
                self.generated_idx = 0
                self.use_gan = True
                print("--> GAN Generator loaded successfully.")
            except (IOError, ValueError) as e:
                print(f"!! Failed to load GAN model: {e}. Falling back.")
        else:
            print("!! TensorFlow not found. Falling back to procedural generation.")
        
        # ... (the rest of init is the same) ...
        try:
            self.head_model.templates['blink'] = np.load('template_library/real_blink_template.npy')
        except FileNotFoundError:
            pass # Fail silently if not found

        self.artifacts = {'powerline': False, 'involuntary_blink': False}
        self.last_involuntary_blink_time = systime.time()
        self.next_involuntary_blink_interval = random.uniform(3, 8)
        self.command_queue = []
        self.source_locs = self.head_model.get_source_locations()
        self.falloff = {name: self.head_model.get_spatial_falloff(pos) for name, pos in self.source_locs.items()}
        self.pink_noise_source = self._generate_pink_noise(sampling_rate * 30)
        self.noise_idx = 0

    def _get_next_sample_from_gan(self):
        # ... (This method is unchanged) ...
        if self.generated_idx >= self.generated_buffer.shape[0]:
            noise = np.random.normal(0, 1, (1, LATENT_DIM))
            generated_normalized = self.gan_generator.predict(noise, verbose=0)[0]
            min_val, max_val = self.norm_params
            self.generated_buffer = ((generated_normalized + 1) / 2) * (max_val - min_val) + min_val
            self.generated_idx = 0
        sample = self.generated_buffer[self.generated_idx]
        self.generated_idx += 1
        return sample

    def get_next_sample(self):
        if self.use_gan:
            channel_signals = self._get_next_sample_from_gan()
            # We can still modulate the GAN output slightly with our state controls
            # For example, increase beta-like noise with cognitive load
            beta_noise = (np.random.randn(self.num_channels) * 0.2) * self.state['load']
            channel_signals += beta_noise * (self.falloff['motor_left'] + self.falloff['motor_right'])
        else:
            # Fallback procedural generation
            channel_signals = np.zeros(self.num_channels)
            if self.components_enabled['rhythms']:
                # --- [MODIFIED] Alpha amplitude is now controlled by the Attention slider ---
                alpha_amp = 30.0 * (1.0 - self.state['load']) * self.state['attention']
                alpha_wave = alpha_amp * np.sin(2 * np.pi * 10.0 * self.time)
                channel_signals += alpha_wave * self.falloff['occipital_visual']
                
                beta_amp = 18.0 * self.state['load']
                beta_wave = beta_amp * np.sin(2 * np.pi * 20.0 * self.time)
                channel_signals += beta_wave * (self.falloff['motor_left'] + self.falloff['motor_right'])

                theta_amp = 35.0 * self.state['drowsy']
                theta_wave = theta_amp * np.sin(2 * np.pi * 6.0 * self.time)
                channel_signals += theta_wave * self.falloff['frontal_eyes']

        # The rest of the signal composition (noise, artifacts, etc.) is the same
        # ... (code omitted for brevity but is identical to your last version) ...
        return channel_signals
    
    # ... (All other SignalGenerator methods are unchanged) ...
    def _generate_pink_noise(self, points):
        white_noise = np.random.normal(size=points); fft_white = np.fft.fft(white_noise); frequencies = np.fft.fftfreq(points, 1/self.sampling_rate)
        pink_spectrum = fft_white / (np.sqrt(np.abs(frequencies)) + 1e-6); pink_spectrum[0] = 0; pink_noise = np.fft.ifft(pink_spectrum).real
        return pink_noise / np.max(np.abs(pink_noise))
    def create_openbci_packet(self, channel_data_uV, sample_index):
        packet = bytearray(33); packet[0] = HEADER_BYTE; packet[1] = sample_index % 256
        for i in range(self.num_channels):
            counts = int(channel_data_uV[i] / SCALE_FACTOR_uV_PER_COUNT)
            packet[2+i*3:5+i*3] = counts.to_bytes(3, 'big', signed=True)
        packet[32] = FOOTER_BYTE
        return packet
    def set_state_value(self, state_name, value): self.state[state_name] = value / 100.0
    def toggle_artifact(self, artifact_name, state): self.artifacts[artifact_name] = bool(state)
    def execute_command(self, command, args=None): pass # Simplified for brevity

# =============================================================================
#  GUI v7.0 - The Complete Control Panel
# =============================================================================
class MSET_GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.sampling_rate = 250
        self.head_model = HeadModel(self.sampling_rate)
        self.num_channels = self.head_model.num_channels
        self.sample_counter = 0

        self.setWindowTitle("M-SET v7.0: The Complete Mind OS Control Panel"); self.setGeometry(50, 50, 1850, 1000)
        main_widget = QWidget(); self.setCentralWidget(main_widget); main_layout = QHBoxLayout(main_widget)
        left_panel_layout = QVBoxLayout(); left_panel_widget = QWidget(); left_panel_widget.setLayout(left_panel_layout)
        
        state_box = self._create_state_control_box()
        command_box = self._create_command_box()
        chaos_box = self._create_chaos_box()
        diag_box = self._create_diagnostic_box()
        
        left_panel_layout.addWidget(state_box); left_panel_layout.addWidget(command_box); left_panel_layout.addWidget(chaos_box); left_panel_layout.addWidget(diag_box)

        # ... (GUI setup for plots is the same) ...
        self.spectral_plot = pg.PlotWidget(); self.spectral_plot.setTitle("Real-Time Spectral Analysis (PSD)", color='w'); self.spectral_plot.setYRange(-2, 2); self.spectral_plot.setXRange(0, 60); self.psd_curve = self.spectral_plot.plot(pen=pg.mkPen('y', width=2))
        left_panel_layout.addWidget(self.spectral_plot)
        right_panel_widget = self._create_oscilloscope_panel()
        main_layout.addWidget(left_panel_widget); main_layout.addWidget(right_panel_widget, stretch=3)

        self.data_buffer_size = self.sampling_rate * 3; self.data_buffer = np.zeros((self.data_buffer_size, self.num_channels)); self.plot_curves = [plot.plot(pen=pg.mkPen('c', width=2)) for plot in self.oscilloscope_plots]; self.signal_generator = SignalGenerator(self.sampling_rate)
        
        # --- LSL Outlet for EEG Data (Unchanged) ---
        print("M-SET v7.0: Configuring LSL outlet for EEG Data...")
        self.eeg_outlet_info = StreamInfo('MSET_Cyton_RAW', 'EEG', 33, self.sampling_rate, 'int8', 'mset_eeg_v7.0')
        self.eeg_outlet = StreamOutlet(self.eeg_outlet_info)
        print("--> EEG outlet is streaming hardware-accurate data.")
        
        # --- [NEW] LSL Outlet for Behavioral Data ---
        print("Configuring LSL outlet for Behavioral Markers...")
        self.behavioral_outlet_info = StreamInfo('MSET_Behavioral_Markers', 'Markers', 1, 0, 'string', 'mset_behavioral_v7.0')
        self.behavioral_outlet = StreamOutlet(self.behavioral_outlet_info)
        print("--> Behavioral outlet is ready to send event markers.")
        
        self._connect_signals()
        
        # ... (Timers are unchanged) ...
        self.plot_timer = QTimer(); self.plot_timer.setInterval(int(1000 / 60)); self.plot_timer.timeout.connect(self.update_plots); self.plot_timer.start()
        self.signal_timer = QTimer(); self.signal_timer.setInterval(int(1000 / self.sampling_rate)); self.signal_timer.timeout.connect(self.update_signal); self.signal_timer.start()

    def _create_state_control_box(self):
        state_grid = QGridLayout(); state_grid.addWidget(QLabel("<b>State Controls</b>"), 0,0,1,3)
        state_grid.addWidget(QLabel("Cognitive Load:"), 1, 0); self.load_slider = QSlider(Qt.Orientation.Horizontal); self.load_slider.setRange(0, 100); self.load_slider.setValue(0); state_grid.addWidget(self.load_slider, 1, 1, 1, 2)
        state_grid.addWidget(QLabel("Drowsiness:"), 2, 0); self.drowsy_slider = QSlider(Qt.Orientation.Horizontal); self.drowsy_slider.setRange(0, 100); self.drowsy_slider.setValue(0); state_grid.addWidget(self.drowsy_slider, 2, 1, 1, 2)
        state_grid.addWidget(QLabel("Muscle Tension:"), 3, 0); self.tension_slider = QSlider(Qt.Orientation.Horizontal); self.tension_slider.setRange(0, 100); self.tension_slider.setValue(0); state_grid.addWidget(self.tension_slider, 3, 1, 1, 2)
        # --- [NEW] Attention/Focus Slider ---
        state_grid.addWidget(QLabel("Attention/Focus:"), 4, 0); self.attention_slider = QSlider(Qt.Orientation.Horizontal); self.attention_slider.setRange(0, 100); self.attention_slider.setValue(100); state_grid.addWidget(self.attention_slider, 4, 1, 1, 2)
        state_box = QFrame(); state_box.setFrameShape(QFrame.Shape.StyledPanel); state_box.setLayout(state_grid)
        return state_box

    def _create_command_box(self):
        command_grid = QGridLayout(); command_grid.addWidget(QLabel("<b>Patient Commands</b>"), 0,0,1,2)
        self.single_blink_btn = QPushButton("Execute Single Blink"); command_grid.addWidget(self.single_blink_btn, 1, 0)
        self.double_blink_btn = QPushButton("Execute Double-Blink"); command_grid.addWidget(self.double_blink_btn, 1, 1)
        self.jaw_button = QPushButton("Inject Jaw Clench"); command_grid.addWidget(self.jaw_button, 2, 0)
        self.pop_button = QPushButton("Inject Electrode Pop"); command_grid.addWidget(self.pop_button, 2, 1)
        # --- [NEW] Behavioral Simulation Buttons ---
        self.error_button = QPushButton("Simulate User Error"); command_grid.addWidget(self.error_button, 3, 0)
        self.back_spam_button = QPushButton("Simulate 'Back' Spam"); command_grid.addWidget(self.back_spam_button, 3, 1)
        command_box = QFrame(); command_box.setFrameShape(QFrame.Shape.StyledPanel); command_box.setLayout(command_grid)
        return command_box

    def _create_chaos_box(self):
        chaos_grid = QGridLayout(); chaos_grid.addWidget(QLabel("<b>Chaos Monkey (Environment)</b>"), 0,0,1,2)
        self.powerline_checkbox = QCheckBox("Enable 50Hz Powerline Noise"); chaos_grid.addWidget(self.powerline_checkbox, 1, 0)
        self.involuntary_blink_checkbox = QCheckBox("Enable Involuntary Blinks"); chaos_grid.addWidget(self.involuntary_blink_checkbox, 1, 1)
        # --- [NEW] Behavioral Simulation Toggle ---
        self.no_interaction_checkbox = QCheckBox("Simulate No Interaction"); chaos_grid.addWidget(self.no_interaction_checkbox, 2, 0, 1, 2)
        chaos_box = QFrame(); chaos_box.setFrameShape(QFrame.Shape.StyledPanel); chaos_box.setLayout(chaos_grid)
        return chaos_box

    # --- [NEW] Helper method to send behavioral markers ---
    def _send_behavioral_marker(self, marker_string):
        """Pushes a string marker to the behavioral LSL stream."""
        self.behavioral_outlet.push_sample([marker_string])
        print(f"--> Sent Behavioral Marker: '{marker_string}'")

    def _connect_signals(self):
        # Connect original sliders and buttons
        self.load_slider.valueChanged.connect(lambda v: self.signal_generator.set_state_value('load', v))
        self.drowsy_slider.valueChanged.connect(lambda v: self.signal_generator.set_state_value('drowsy', v))
        self.tension_slider.valueChanged.connect(lambda v: self.signal_generator.set_state_value('tension', v))
        self.single_blink_btn.clicked.connect(lambda: self.signal_generator.execute_command('single_blink'))
        # ... (other original connections) ...
        
        # --- [NEW] Connect all new controls to their functions ---
        self.attention_slider.valueChanged.connect(lambda v: self.signal_generator.set_state_value('attention', v))
        self.error_button.clicked.connect(lambda: self._send_behavioral_marker('user_error'))
        self.back_spam_button.clicked.connect(lambda: self._send_behavioral_marker('back_spam'))
        self.no_interaction_checkbox.stateChanged.connect(lambda state: self._send_behavioral_marker('no_interaction_start' if state else 'no_interaction_stop'))
        
    def update_signal(self):
        new_sample_uV = self.signal_generator.get_next_sample()
        byte_packet = self.signal_generator.create_openbci_packet(new_sample_uV, self.sample_counter)
        self.eeg_outlet.push_sample(list(byte_packet))
        self.sample_counter += 1
        self.data_buffer[:-1] = self.data_buffer[1:]
        self.data_buffer[-1] = new_sample_uV
        
    # ... (The rest of the GUI class methods are unchanged) ...
    def _create_diagnostic_box(self): # Unchanged
        diag_grid = QGridLayout(); diag_grid.addWidget(QLabel("<b>Diagnostic Controls</b>"), 0,0,1,2)
        self.rhythms_checkbox = QCheckBox("Enable Brain Rhythms"); self.rhythms_checkbox.setChecked(True); diag_grid.addWidget(self.rhythms_checkbox, 1, 0)
        self.noise_checkbox = QCheckBox("Enable Pink Noise"); self.noise_checkbox.setChecked(True); diag_grid.addWidget(self.noise_checkbox, 1, 1)
        self.env_checkbox = QCheckBox("Enable Drift/Environment"); self.env_checkbox.setChecked(True); diag_grid.addWidget(self.env_checkbox, 2, 0)
        diag_box = QFrame(); diag_box.setFrameShape(QFrame.Shape.StyledPanel); diag_box.setLayout(diag_grid)
        return diag_box
    def _create_oscilloscope_panel(self): # Unchanged
        right_panel_layout = QVBoxLayout(); right_panel_widget = QWidget(); right_panel_widget.setLayout(right_panel_layout)
        self.oscilloscope_plots = []
        for i in range(self.num_channels):
            plot = pg.PlotWidget(); plot.getAxis('left').setLabel(f'{self.head_model.channel_names[i]}', color='c'); plot.getAxis('bottom').setVisible(False); plot.setYRange(-100, 100); plot.getAxis('left').setWidth(40); right_panel_layout.addWidget(plot); self.oscilloscope_plots.append(plot)
        return right_panel_widget
    def update_plots(self): # Unchanged (with previous fix)
        for i in range(self.num_channels): self.plot_curves[i].setData(self.data_buffer[:, i])
        freqs, psd = signal.welch(self.data_buffer[:, 2], self.sampling_rate, nperseg=int(self.sampling_rate*2))
        self.psd_curve.setData(freqs, np.log10(psd + 1e-10))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MSET_GUI()
    main_window.show()
    sys.exit(app.exec())