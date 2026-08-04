import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l1
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# Create output folder
results = []
output_folder = "new_models"
os.makedirs(output_folder, exist_ok=True)

# Load the dataset
import pandas as pd

# Load the training data
X_train = pd.read_csv('/Users/gouthamkrishnan/Downloads/human+activity+recognition+using+smartphones/UCI HAR Dataset/train/X_train.txt', delim_whitespace=True, header=None)
y_train = pd.read_csv('/Users/gouthamkrishnan/Downloads/human+activity+recognition+using+smartphones/UCI HAR Dataset/train/y_train.txt', delim_whitespace=True, header=None)

# Load the test data
X_test = pd.read_csv('/Users/gouthamkrishnan/Downloads/human+activity+recognition+using+smartphones/UCI HAR Dataset/test/X_test.txt', delim_whitespace=True, header=None)
y_test = pd.read_csv('/Users/gouthamkrishnan/Downloads/human+activity+recognition+using+smartphones/UCI HAR Dataset/test/y_test.txt', delim_whitespace=True, header=None)

# Merge train and test sets
X = pd.concat([X_train, X_test])
y = pd.concat([y_train, y_test])

# print(f'Dataset shape: {X.shape}, Labels shape: {y.shape}')
# Separate features and target
# X = train_data.iloc[:, :-1].values
# y = train_data['quality'].values

# Normalize features to range (0, 0.1)
scaler = MinMaxScaler(feature_range=(0, 0.1))
X_scaled = scaler.fit_transform(X)
#
# # Shift target labels to start from 0 (quality 3 to 9 → 0 to 6)
y_new= y-1
num_classes = len(np.unique(y_train))
#
# # Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_new, test_size=0.2, random_state=42)
#
# # One-hot encode the target
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)
#
# Define clipped ReLU activation
def clipped_relu(x):
    return K.relu(x, max_value=0.1)

# Build the model
model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation=clipped_relu),

    Dense(32, activation=clipped_relu,kernel_regularizer=l1(1e-5)),

    Dense(num_classes, activation='sigmoid', kernel_regularizer=l1(1e-5))  # softmax for multiclass
])
#
# # Compile the model
model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
#
# # Train the model
history = model.fit(X_train, y_train_cat, epochs=100, batch_size=32, validation_split=0.1, verbose=1)
#
# # Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}, loss : {loss:.4f}")
#
# Optional: print classification report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print("\nClassification Report:\n")
# print(classification_report(y_test, y_pred_classes))
#
# # Optional: plot accuracy
# plt.plot(history.history['accuracy'], label='train')
# plt.plot(history.history['val_accuracy'], label='val')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.title('Model Accuracy over Epochs')
# plt.show()
# model.save("new_model.h5")
# print("Model saved successfully!")
# for layers in model.layers :
#     print(layers)
# # Get original weights
# cm = confusion_matrix(y_test, y_pred_classes)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(1, num_classes+1),
#             yticklabels=np.arange(1, num_classes+1))
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.show()
#
# # Plot weight distributions for each Dense layer
# for i, layer in enumerate(model.layers):
#     if isinstance(layer, Dense):
#         weights, biases = layer.get_weights()
#         plt.figure(figsize=(6, 4))
#         plt.hist(weights.flatten(), bins=50, color='green', alpha=0.7)
#         plt.title(f'Weight Distribution - Layer {i} ({layer.name})')
#         plt.xlabel("Weight Value")
#         plt.ylabel("Frequency")
#         plt.grid(True)
#         plt.show()

weights_1 = model.layers[0].get_weights()[0]
weights_2 = model.layers[1].get_weights()[0]
weights_3 = model.layers[2].get_weights()[0]

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
layers = [weights_1, weights_2, weights_3]
titles = ['Layer 1 Weights (Input → Dense1)',
          'Layer 2 Weights (Dense1 → Dense2)',
          'Layer 3 Weights (Dense2 → Output)']

for i, (weights, title) in enumerate(zip(layers, titles)):
    axs[i].hist(weights.flatten(), bins=50, color='skyblue', edgecolor='black')
    axs[i].set_title(title)
    axs[i].set_xlabel("Weight Value")
    axs[i].set_ylabel("Frequency")

plt.tight_layout()
plt.show()











