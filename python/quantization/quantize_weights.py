import tensorflow as tf
from keras.src.utils import to_categorical
from qkeras.quantizers import quantized_bits
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


import os
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler
results = []
output_folder = "Q_new_models"
os.makedirs(output_folder, exist_ok=True)

X_train = pd.read_csv('/Users/gouthamkrishnan/Downloads/human+activity+recognition+using+smartphones/UCI HAR Dataset/train/X_train.txt', delim_whitespace=True, header=None)
y_train = pd.read_csv('/Users/gouthamkrishnan/Downloads/human+activity+recognition+using+smartphones/UCI HAR Dataset/train/y_train.txt', delim_whitespace=True, header=None)

# Load the test data
X_test = pd.read_csv('/Users/gouthamkrishnan/Downloads/human+activity+recognition+using+smartphones/UCI HAR Dataset/test/X_test.txt', delim_whitespace=True, header=None)
y_test = pd.read_csv('/Users/gouthamkrishnan/Downloads/human+activity+recognition+using+smartphones/UCI HAR Dataset/test/y_test.txt', delim_whitespace=True, header=None)

# Merge train and test sets
X = pd.concat([X_train, X_test])
y = pd.concat([y_train, y_test])

# Normalize features to range (0, 0.1)
scaler = MinMaxScaler(feature_range=(0, 0.1))
X_scaled = scaler.fit_transform(X)
#
# # Shift target labels to start from 0 (quality 3 to 9 → 0 to 6)
y_new= y-1
num_classes = len(np.unique(y_train))  # Should be 7
#
# # Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_new, test_size=0.2, random_state=42)
#
# # One-hot encode the target
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

def clipped_relu(x):
    return K.relu(x, max_value=0.1)

print(X_train.shape[1])
# #Grid Search for q_levels determination
q_levels=[1,2,3,4,5,6]

def quantize_weights_and_biases(weights, biases, weight_levels, bias_levels):
    quantized_weights = quantized_bits(weight_levels)(weights)
    quantized_biases = quantized_bits(bias_levels)(biases)
    return quantized_weights, quantized_biases

for w1_i in q_levels:
    for b1_i in q_levels:
        for w2_i in q_levels:
            for b2_i in q_levels:
                for w3_i in q_levels:
                    for b3_i in q_levels:
                        model = load_model("/Users/gouthamkrishnan/PycharmProjects/multplication/new_model.h5", custom_objects={"clipped_relu": clipped_relu})

                        # Get original weights
                        w0, b0 = model.layers[0].get_weights()
                        w1, b1 = model.layers[1].get_weights()
                        w2, b2 = model.layers[2].get_weights()

                        # Quantize
                        q_w0, q_b0 = quantize_weights_and_biases(w0, b0, w1_i, b1_i)
                        q_w1, q_b1 = quantize_weights_and_biases(w1, b1, w2_i, b2_i)
                        q_w2, q_b2 = quantize_weights_and_biases(w2, b2, w3_i, b3_i)

                        # Set quantized weights
                        model.layers[0].set_weights([q_w0, q_b0])
                        model.layers[1].set_weights([q_w1, q_b1])
                        model.layers[2].set_weights([q_w2, q_b2])

                        # Evaluate
                        test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, batch_size=32, verbose=0)

                        # Save
                        model_name = f"q_l1w{w1_i}_l1b{b1_i}_l2w{w2_i}_l2b{b2_i}_l3w{w3_i}_l3b{b3_i}.h5"
                        model_path = os.path.join(output_folder, model_name)
                        model.save(model_path)

                        results.append({
                            "L1_Weight": w1_i, "L1_Bias": b1_i,
                            "L2_Weight": w2_i, "L2_Bias": b2_i,
                            "L3_Weight": w3_i, "L3_Bias": b3_i,
                            "Loss": test_loss, "Accuracy": test_accuracy,
                            "Model": model_name
                        })


results_df = pd.DataFrame(results)
results_df.to_csv("q_results_HAR.csv", index=False)