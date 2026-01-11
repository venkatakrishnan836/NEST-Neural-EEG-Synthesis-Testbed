#
# --- FILE: train_gan.py (v4 - With Improved Progress Logging) ---
#
import tensorflow as tf
import numpy as np
import os
from build_gan import build_generator, build_critic

# --- Hyperparameters ---
EPOCHS = 20000
BATCH_SIZE = 64
LEARNING_RATE = 0.0002
BETA_1 = 0.5
LATENT_DIM = 128
N_CRITIC = 5
GP_WEIGHT = 10.0

# --- Load Data ---
print("Loading processed dataset...")
real_eeg_data = np.load('als_eeg_processed_dataset.npy')
real_eeg_data = real_eeg_data.astype(np.float32)

# Add .repeat() to make the dataset loop indefinitely
train_dataset_iterator = iter(tf.data.Dataset.from_tensor_slices(real_eeg_data).shuffle(real_eeg_data.shape[0]).batch(BATCH_SIZE, drop_remainder=True).repeat())

print(f"Dataset loaded with {real_eeg_data.shape[0]} samples. Data type is now {real_eeg_data.dtype}. Dataset will now repeat.")

# --- Build Models and Optimizers ---
generator = build_generator()
critic = build_critic()

g_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1)
c_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1)

# (Loss functions and train_step are unchanged)
def critic_loss(real_output, fake_output):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)

def gradient_penalty(batch_size, real_eeg, fake_eeg):
    alpha = tf.random.uniform([batch_size, 1, 1])
    interpolated = real_eeg + alpha * (fake_eeg - real_eeg)
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = critic(interpolated, training=True)
    grads = gp_tape.gradient(pred, [interpolated])[0]
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

@tf.function
def train_step(real_eeg):
    batch_size = tf.shape(real_eeg)[0]
    for _ in range(N_CRITIC):
        with tf.GradientTape() as c_tape:
            noise = tf.random.normal([batch_size, LATENT_DIM])
            fake_eeg = generator(noise, training=True)
            real_output = critic(real_eeg, training=True)
            fake_output = critic(fake_eeg, training=True)
            c_loss = critic_loss(real_output, fake_output)
            gp = gradient_penalty(batch_size, real_eeg, fake_eeg)
            total_c_loss = c_loss + gp * GP_WEIGHT
        c_gradients = c_tape.gradient(total_c_loss, critic.trainable_variables)
        c_optimizer.apply_gradients(zip(c_gradients, critic.trainable_variables))
    with tf.GradientTape() as g_tape:
        noise = tf.random.normal([batch_size, LATENT_DIM])
        fake_eeg = generator(noise, training=True)
        fake_output = critic(fake_eeg, training=True)
        g_loss = generator_loss(fake_output)
    g_gradients = g_tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
    return total_c_loss, g_loss

# --- The Main Training Loop ---
print("Starting GAN training...")
os.makedirs("gan_models", exist_ok=True)

steps_per_epoch = real_eeg_data.shape[0] // BATCH_SIZE

for epoch in range(EPOCHS):
    for step in range(steps_per_epoch):
        eeg_batch = next(train_dataset_iterator)
        c_loss, g_loss = train_step(eeg_batch)
    
    # ===================================================================
    # THE MODIFIED LOGGING AND SAVING LOGIC
    # ===================================================================
    # Check if the current epoch is a multiple of 100 (for saving)
    if (epoch + 1) % 100 == 0:
        # On the 100th, 200th, etc. epoch, print a detailed summary and save the model.
        # This print will be on a new line and will stay in your log.
        print(f"\nEpoch {epoch+1}/{EPOCHS}, Critic Loss: {c_loss.numpy():.4f}, Generator Loss: {g_loss.numpy():.4f}")
        generator.save(f"gan_models/generator_epoch_{epoch+1}.h5")
    else:
        # For all other epochs (1-99, 101-199, etc.), just update the status on a single line.
        # The `end='\r'` moves the cursor to the beginning of the line without starting a new one.
        print(f"Processing Epoch: {epoch + 1}/{EPOCHS}", end='\r')
    # ===================================================================

print("\nTraining finished.")