

# Write a program to evaluate a Recurrent Neural Network (RNN) for text Classification.  DAta set Name is 
############# BBC_news Data set ################

<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
# Step 1: Import libraries
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 3: Define dataset path
dataset_path = '/content/drive/MyDrive/bbc_news'  # contains subfolders like tech, sport, etc.

# Step 4: Load texts and labels from folders
texts = []
labels = []

categories = os.listdir(dataset_path)
print("Categories:", categories)

for category in categories:
    folder_path = os.path.join(dataset_path, category)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='latin1') as f:
            text = f.read()
            texts.append(text)
            labels.append(category)

print(f"Loaded {len(texts)} documents.")

# Step 5: Encode labels to integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Step 6: Tokenize texts
max_words = 10000  # max vocabulary size
max_len = 200      # max length of each sequence

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=max_len, padding='post')

print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# Step 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Build RNN model
embedding_dim = 64

model = Sequential([
    Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len),
    SimpleRNN(64, return_sequences=False),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(len(categories), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Step 9: Train the model
epochs = 10
history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.1)

# Step 10: Evaluate model on test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy*100:.2f}%")

# Step 11: Example prediction
sample_text = ["The government is planning new economic reforms."]

sample_seq = tokenizer.texts_to_sequences(sample_text)
sample_pad = pad_sequences(sample_seq, maxlen=max_len, padding='post')

pred = model.predict(sample_pad)
pred_class = label_encoder.inverse_transform([np.argmax(pred)])

print(f"Prediction for sample text: {pred_class[0]}")

# Step 12: Plot Accuracy & Loss Graphs
plt.figure(figsize=(12,5))

# Accuracy plot
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='green')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='blue')
plt.title("Accuracy over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Loss plot
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss', color='red')
plt.plot(history.history['val_loss'], label='Val Loss', color='orange')
plt.title("Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()
