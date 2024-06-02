import csv
import random
import numpy as np
from nn import MLP, Value

def normalize_input(X):
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    return (X - means) / stds 

# Load and preprocess the data
with open('wine.csv', 'r') as file:
    reader = csv.reader(file)
    header = next(reader)
    rows = []
    for row in reader:
        row = [float(value) if i > 0 else int(value) - 1 for i, value in enumerate(row)]  # Convert labels to 0-based index
        rows.append(row)

# Shuffle and split the data into train and test sets
random.shuffle(rows)
split = int(0.9 * len(rows))
train_rows = rows[:split]
test_rows = rows[split:]

X_train = [row[1:] for row in train_rows]  # Features
y_train = [row[0] for row in train_rows]  # Labels
X_test = [row[1:] for row in test_rows]
y_test = [row[0] for row in test_rows]

# Normalize the input data
X_train = normalize_input(np.array(X_train))
X_test = normalize_input(np.array(X_test))

# Create the model
model = MLP(13, [32, 16, 3]) 

# Define training parameters
iterations = 3000 #100000
lr = 0.001
grad_accumulations = 10 #2500

# Train and evaluate the model
for i in range(iterations):
    for k in range(grad_accumulations):
        # Select a random training example
        sample = train_rows[i % len(train_rows)]
        x = sample[0]
        y_label = sample[1]
        # Forward pass
        input_vec = [Value(xi) for xi in x.tolist()]
        y_pred = model(input_vec)
        # Compute probabilities
        probs = Value.softmax(y_pred)
        # Compute cross-entropy loss
        loss = -1 * probs[y_label].log() / float(grad_accumulations)

        # Backward pass
        model.zero_grad()
        loss.backward()
        # Update weights - SGD (Stochastic Gradient Descent)
        for param in model.parameters():
            param.data -= lr * param.grad
        
    if i % 1000 == 0:
        print(f'Iteration {i}, Loss: {loss.data}')

correct = 0
for input_vec, target in zip(X_test, y_test):
    output = Value.softmax(model([Value(x) for x in input_vec]))
    predicted = max(range(len(output)), key=lambda i: output[i].data)
    if predicted == target:
        correct += 1
accuracy = correct / len(X_test)
print(f"Accuracy: {accuracy:.4f}")