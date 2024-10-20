# Step 1: Import required libraries
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Step 2: Define XOR input and output (truth table)
# Inputs: (X1, X2) and Outputs: XOR(X1, X2)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # XOR inputs
y = np.array([[0], [1], [1], [0]])  # XOR outputs

# Step 3: Create a Multilayer Perceptron (MLP) model
model = Sequential()

# Input layer with 2 neurons (for 2 inputs) and one hidden layer with 2 neurons
model.add(Dense(2, input_dim=2, activation='relu'))  # Hidden layer
# Output layer with 1 neuron (for the XOR output)
model.add(Dense(1, activation='sigmoid'))  # Output layer

# Step 4: Compile the model
# Binary Crossentropy is used because XOR is a binary classification problem
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 5: Train the model
# Training the model with the XOR input and output for 500 epochs
history = model.fit(X, y, epochs=500, verbose=0)

# Step 6: Evaluate the model performance
# Test the model on the XOR inputs
print("\nModel evaluation:")
_, accuracy = model.evaluate(X, y, verbose=0)
print(f'Accuracy: {accuracy*100:.2f}%')

# Step 7: Make predictions on XOR inputs
predictions = model.predict(X)

# Step 8: Display the results
# Threshold the predictions to 0 or 1 to simulate XOR logic
thresholded_predictions = (predictions > 0.5).astype(int)

# Print input, predicted output, and actual output for comparison
print("\nXOR Gate Results:")
for i in range(len(X)):
    print(f"Input: {X[i]} - Predicted Output: {thresholded_predictions[i]} - Actual Output: {y[i]}")
