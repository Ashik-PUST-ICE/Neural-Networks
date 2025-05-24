

# Write a program to evaluate a Recurrent Neural Network (RNN) for text Classification.  DAta set Name is 
############# user-data ################


# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam

# Step 2: Load CSV File
df = pd.read_csv('/content/user-data.csv')

# Step 3: Data Preprocessing
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})  # Gender to numeric
df = df.drop('user_id', axis=1)  # Remove user_id
X = df.drop('purchased', axis=1)
y = df['purchased']

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# RNN needs 3D input: (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Step 6: Build RNN Model
model = Sequential()
model.add(SimpleRNN(32, input_shape=(3,1)))
model.add(Dense(1, activation='sigmoid'))  # Output: 1 neuron for binary classification

# Step 7: Compile the Model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Step 8: Train the Model
history = model.fit(X_train, y_train, epochs=50, verbose=1)

# Step 9: Evaluate Model on Test Data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Step 10: Plot Accuracy & Loss Curve
plt.figure(figsize=(12,5))

# Accuracy Plot
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Accuracy', color='green')
plt.title("Training Accuracy over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Loss Plot
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Loss', color='red')
plt.title("Training Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

# Step 11: Random New User Prediction
# Example: Male, 30 years old, salary 80000
new_user = np.array([[1, 40, 120000]])  # 2D
new_user_scaled = scaler.transform(new_user)
new_user_scaled = new_user_scaled.reshape((1, 3, 1))  # Reshape for RNN

prediction = model.predict(new_user_scaled)
print(f"\nPrediction for New User: {prediction[0][0]:.4f}")

if prediction[0][0] >= 0.5:
    print("→ Model Predicts: This user will PURCHASE ✅")
else:
    print("→ Model Predicts: This user will NOT purchase ❌")
