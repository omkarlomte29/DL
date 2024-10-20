# Exploring Deep Learning Libraries in Python

# Step 1: Importing Libraries
import theano
import tensorflow as tf
from tensorflow import keras

print("Theano Version:", theano.__version__)
print("TensorFlow Version:", tf.__version__)
print("Keras Version:", keras.__version__)

# Step 2: Define a Simple Neural Network using TensorFlow and Keras
# Define a simple Sequential model using Keras (which runs on top of TensorFlow)
model = keras.Sequential([
    # Input layer with 2 neurons
    keras.layers.Dense(2, activation='relu', input_shape=(2,)),
    
    # Hidden layer with 4 neurons
    keras.layers.Dense(4, activation='relu'),
    
    # Output layer with 1 neuron
    keras.layers.Dense(1, activation='sigmoid')
])

# Step 3: Compile the Model
# Use binary cross-entropy loss and adam optimizer
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display model summary to understand the architecture
model.summary()

# Step 4: Example Dataset (For XOR Problem)
# This dataset simulates the XOR gate logic
import numpy as np
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Step 5: Train the Model
# Train the model for 1000 epochs with the XOR dataset
history = model.fit(X, y, epochs=1000, verbose=0)

# Step 6: Evaluate the Model
print("\nModel Evaluation:")
model.evaluate(X, y)

# Step 7: Make Predictions
print("\nModel Predictions:")
predictions = model.predict(X)
thresholded_predictions = (predictions > 0.5).astype(int)
for i in range(len(X)):
    print(f"Input: {X[i]}, Predicted Output: {thresholded_predictions[i][0]}, Actual Output: {y[i][0]}")
