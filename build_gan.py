#
# --- FILE: build_gan.py (v2 - Corrected for Keras API) ---
#
import tensorflow as tf
# [MODIFIED] Import Cropping1D
from tensorflow.keras.layers import Input, Dense, Reshape, Conv1D, Conv1DTranspose, Flatten, LeakyReLU, BatchNormalization, Cropping1D
from tensorflow.keras.models import Model

# --- Hyperparameters ---
LATENT_DIM = 128
EEG_SHAPE = (250, 8) # 250 samples, 8 channels

def build_critic():
    """Builds the Critic (Discriminator) model."""
    input_eeg = Input(shape=EEG_SHAPE)
    
    # [MODIFIED] Changed deprecated 'alpha' to 'negative_slope'
    x = Conv1D(64, kernel_size=5, strides=2, padding='same')(input_eeg)
    x = LeakyReLU(negative_slope=0.2)(x)
    
    x = Conv1D(128, kernel_size=5, strides=2, padding='same')(x)
    x = LeakyReLU(negative_slope=0.2)(x)

    x = Conv1D(256, kernel_size=5, strides=2, padding='same')(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    
    x = Flatten()(x)
    x = Dense(1)(x)
    
    model = Model(input_eeg, x, name="critic")
    return model

def build_generator():
    """Builds the Generator model."""
    latent_input = Input(shape=(LATENT_DIM,))
    
    x = Dense(32 * 256, use_bias=False)(latent_input)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x) # Fixed alpha
    x = Reshape((32, 256))(x)

    x = Conv1DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x) # Fixed alpha

    x = Conv1DTranspose(64, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x) # Fixed alpha

    x = Conv1DTranspose(32, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x) # Fixed alpha
    # At this point, the shape is (None, 256, 32)
    
    # ===================================================================
    # THE FIX: Replace tf.slice with the proper Keras Cropping1D layer
    # To get from 256 samples to 250, we need to crop 6 total samples.
    # We will crop 3 from the beginning and 3 from the end.
    # ===================================================================
    # [REPLACED] x = tf.slice(x, [0, 0, 0], [-1, 250, -1])
    x = Cropping1D(cropping=(3, 3))(x) # New shape is (None, 250, 32)
    # ===================================================================
    
    # Final layer to get to 8 channels with a tanh activation
    output_eeg = Conv1D(EEG_SHAPE[1], kernel_size=7, padding='same', activation='tanh')(x)

    model = Model(latent_input, output_eeg, name="generator")
    return model

# --- Test the builders ---
if __name__ == '__main__':
    generator = build_generator()
    critic = build_critic()
    
    print("--- Generator Summary ---")
    generator.summary()
    
    print("\n--- Critic Summary ---")
    critic.summary()