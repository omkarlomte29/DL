import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# Load the MNIST dataset (you can replace this with CIFAR-10 for color images)
(X_train, _), (X_test, _) = tf.keras.datasets.mnist.load_data()

# Normalize data to [0, 1] range and add a channel dimension
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

# Add a channel dimension for compatibility with convolutional layers
X_train = np.reshape(X_train, (len(X_train), 28, 28, 1))
X_test = np.reshape(X_test, (len(X_test), 28, 28, 1))

# Add noise to the data for the denoising task
noise_factor = 0.5
X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape) 
X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape) 
X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)

# Define the Autoencoder architecture
def build_autoencoder():
    # Encoder
    input_img = layers.Input(shape=(28, 28, 1))
    
    # Encoding layers (reduces the dimensions)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Decoder
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    # Output layer (reconstructs the original image)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    # Define the autoencoder model
    autoencoder = models.Model(input_img, decoded)
    return autoencoder

# Build the autoencoder model
autoencoder = build_autoencoder()

# Compile the model with optimizer and loss function
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder on noisy data (for denoising) or regular data (for compression)
autoencoder.fit(X_train_noisy, X_train, epochs=50, batch_size=256, shuffle=True, validation_data=(X_test_noisy, X_test))

# After training, use the autoencoder to denoise the images
decoded_imgs = autoencoder.predict(X_test_noisy)

# Visualize the results: Original, Noisy, and Denoised images
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # Display noisy
    ax = plt.subplot(3, n, i + n + 1)
    plt.imshow(X_test_noisy[i].reshape(28, 28), cmap='gray')
    plt.title("Noisy")
    plt.axis('off')

    # Display denoised (reconstructed)
    ax = plt.subplot(3, n, i + 2 * n + 1)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    plt.title("Denoised")
    plt.axis('off')

plt.show()
