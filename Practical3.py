# Import the required libraries
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Step 1: Load and preprocess the MNIST dataset
# MNIST dataset contains 28x28 images of handwritten digits 0-9
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the data to the range [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Step 2: One-hot encode the labels (since it's a multi-class classification problem)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Step 3: Define the model (Deep Neural Network)
model = Sequential()

# Step 4: Add layers to the model
# Flatten layer to convert 2D images to 1D vectors (28x28 -> 784)
model.add(Flatten(input_shape=(28, 28)))

# First hidden layer with 128 neurons and ReLU activation
model.add(Dense(128, activation='relu'))

# Second hidden layer with 64 neurons and ReLU activation
model.add(Dense(64, activation='relu'))

# Output layer with 10 neurons (for 10 classes) and softmax activation
model.add(Dense(10, activation='softmax'))

# Step 5: Compile the model
# Adam optimizer and categorical cross-entropy loss function
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Step 6: Train the model
# Fit the model on the training data for 10 epochs with a batch size of 32
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Step 7: Evaluate the model
# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# Step 8: Predict on new data (optional)
# Let's predict the first 5 images from the test set
predictions = model.predict(X_test[:5])

# Show the predictions and the corresponding true labels
for i in range(5):
    print(f"Predicted: {np.argmax(predictions[i])}, True label: {np.argmax(y_test[i])}")

