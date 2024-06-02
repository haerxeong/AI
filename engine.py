#engine.py
import random
from nn import Value
import numpy as np
import csv

def load_and_process_data(file_path):
    with open(file_path, 'r') as file:
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

    return X_train, y_train, X_test, y_test

def normalize_input(X):
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    return (X - means) / stds

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, iterations, learning_rate, grad_accumulations):
    train(model, X_train, y_train, iterations, learning_rate, grad_accumulations)
    evaluate(model, X_test, y_test)

def train(model, X, y, iterations, learning_rate, grad_accumulations):
    rows = list(zip(X, y))
    random.shuffle(rows)
    split = int(0.9 * len(rows))
    train_rows = rows[:split]
    test_rows = rows[split:]

    for i in range(iterations):
        for k in range(grad_accumulations):
            # Select a random training example
            sample = train_rows[i % len(train_rows)]
            x = sample[0]
            y = sample[1]
            # Forward pass
            input_vec = [Value(xi) for xi in x]
            y_pred = model(input_vec)
            # Compute probabilities
            probs = Value.softmax(y_pred)
            # Compute cross-entropy loss
            loss = -1 * probs[y].log() / float(grad_accumulations)

            # Backward pass
            model.zero_grad()
            loss.backward()
            # Update weights - SGD (Stochastic Gradient Descent)
            for param in model.parameters():
                param.data -= learning_rate * param.grad
        
        if i % 1000 == 0:
            print(f'Iteration {i}, Loss: {loss.data}')
        
    count = 0
    for sample in test_rows:
        x = sample[0]
        y = sample[1]
        input_vec = [Value(xi) for xi in x]
        y_pred = model(input_vec)
        probs = Value.softmax(y_pred)
        probs_data = [prob.data for prob in probs]
        if np.argmax(probs_data) == y:
            count += 1
    accuracy = count / len(test_rows)
    print(f'Accuracy: {accuracy:.4f}')

def evaluate(model, X, y):
    correct = 0
    for input_vec, target in zip(X, y):
        output = Value.softmax(model([Value(x) for x in input_vec]))
        predicted = max(range(len(output)), key=lambda i: output[i].data)
        if predicted == target:
            correct += 1
    accuracy = correct / len(X)
    print(f"Accuracy: {accuracy:.4f}")
