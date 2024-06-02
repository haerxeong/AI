import csv
from engine import load_and_process_data, train_and_evaluate_model
from nn import MLP

# Load and preprocess the data
X_train, y_train, X_test, y_test = load_and_process_data('wine.csv')

# Create the MLP model
model = MLP(13, [32, 16, 3])  # Adjusted architecture for 13 input features

# Define training parameters
iterations = 20000
learning_rate = 0.001
grad_accumulations = 10

# Train and evaluate the model
train_and_evaluate_model(model, X_train, y_train, X_test, y_test, iterations, learning_rate, grad_accumulations)