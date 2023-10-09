import tensorflow as tf
from tensorflow import keras
import numpy as np

# Create a sample dataset for binary classification
# Input features
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Target labels
y = np.array([0, 1, 1, 0])

# Build a simple feedforward neural network
model = keras.Sequential([
    keras.layers.Input(shape=(2,)),  # Input layer with 2 input features
    keras.layers.Dense(4, activation='relu'),  # Hidden layer with 4 neurons and ReLU activation
    keras.layers.Dense(1, activation='sigmoid')  # Output layer with 1 neuron and sigmoid activation
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=1000, verbose=0)  # Train for 1000 epochs

# Evaluate the model
loss, accuracy = model.evaluate(X, y)
print(f"Loss: {loss:.4f}, Accuracy: {accuracy * 100:.2f}%")

# Make predictions
predictions = model.predict(X)
print("Predictions:")
for i in range(len(predictions)):
    print(f"Input: {X[i]}, Predicted Probability: {predictions[i][0]:.4f}")
