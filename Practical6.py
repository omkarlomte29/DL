import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load the MNIST dataset from TensorFlow
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the images to [0, 1] range by dividing by 255
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape the data to add a channel dimension (since we are working with grayscale images)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Convert the labels to one-hot encoded format (for multi-class classification)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Define the CNN model architecture
def create_cnn_model():
    model = models.Sequential()
    
    # First Convolutional layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Second Convolutional layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Third Convolutional layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    # Flatten the output to feed into fully connected layers
    model.add(layers.Flatten())
    
    # Fully connected layer
    model.add(layers.Dense(64, activation='relu'))
    
    # Output layer for 10 classes (digits 0-9), using softmax activation
    model.add(layers.Dense(10, activation='softmax'))
    
    return model

# Create the CNN model
cnn_model = create_cnn_model()

# Compile the model using Adam optimizer, cross-entropy loss, and accuracy as the metric
cnn_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Train the CNN model
cnn_model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model on the test data
test_loss, test_acc = cnn_model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Predict the first 10 images in the test set and visualize them
predictions = cnn_model.predict(X_test[:10])

# Plot the first 10 test images and their predicted labels
plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(5, 5, i+1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predictions[i].argmax()}")
    plt.axis('off')
plt.show()
