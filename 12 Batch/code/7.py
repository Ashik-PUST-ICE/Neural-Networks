

#  Write a program to evaluate a Transformer model for text classification. 


# ##### Data set is BBC-news



import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, GlobalAveragePooling1D, LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

# Transformer block definition
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Load your BBC dataset folder path here
dataset_path = '/content/drive/MyDrive/bbc_news'  # Change to your dataset folder

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

# Label encode the categories
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Tokenize texts
max_words = 10000
max_len = 200

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=max_len, padding='post')

print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transformer model parameters
embed_dim = 64  # Embedding size for each token
num_heads = 2   # Number of attention heads
ff_dim = 64     # Hidden layer size in feed-forward network inside transformer

# Build Transformer model
inputs = Input(shape=(max_len,))
embedding_layer = Embedding(max_words, embed_dim)(inputs)

transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(embedding_layer, training=True)  # pass training=True for demo, Keras will manage during fit

x = GlobalAveragePooling1D()(x)
x = Dropout(0.5)(x)
x = Dense(32, activation="relu")(x)
x = Dropout(0.5)(x)
outputs = Dense(len(categories), activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Train the model
epochs = 10
history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.1)

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy*100:.2f}%")

# Example prediction
sample_text = ["The government is planning new economic reforms."]
sample_seq = tokenizer.texts_to_sequences(sample_text)
sample_pad = pad_sequences(sample_seq, maxlen=max_len, padding='post')

pred = model.predict(sample_pad)
pred_class = label_encoder.inverse_transform([np.argmax(pred)])

print(f"Prediction for sample text: {pred_class[0]}")
