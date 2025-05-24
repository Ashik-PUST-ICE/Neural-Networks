

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the MNIST dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target.astype(int)

# Step 2: Normalize and split the dataset
X = X / 255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to DataFrame/Series for iloc access
X_test = pd.DataFrame(X_test)
y_test = pd.Series(y_test)

# Step 3: Initialize and train the model
print("Training the model...")
model = MLPClassifier(hidden_layer_sizes=(128,), activation='relu', solver='adam', max_iter=10, random_state=42)
model.fit(X_train, y_train)

# Step 4: Evaluate the model
print("Evaluating the model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'\n✅ Test accuracy: {accuracy*100:.2f}%')

# Step 5: Function to plot image and prediction
def plot_image(i, true_label, img):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img.values.reshape(28, 28), cmap=plt.cm.binary)
    plt.xlabel(f"True: {true_label}\nPredicted: {y_pred[i]}", color='blue')

# Step 6: Display 100 sample predictions in 10x10 grid
num_rows, num_cols = 10, 10
plt.figure(figsize=(2*num_cols, 2*num_rows))
for i in range(num_rows * num_cols):
    plt.subplot(num_rows, num_cols, i+1)
    plot_image(i, y_test.iloc[i], X_test.iloc[i])
plt.tight_layout()
plt.show()
