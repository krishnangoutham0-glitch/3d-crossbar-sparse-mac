import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from keras.src.ops import binary_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Input,Lambda,Activation, Subtract,Concatenate
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix
import seaborn as sns

def clipped_relu(x):
    return K.relu(x, max_value=0.1)

 

model_path = "/Users/gouthamkrishnan/PycharmProjects/multplication/Q_new_models/q_l1w5_l1b6_l2w5_l2b6_l3w1_l3b3.h5"

# -------- Load and Prepare Data --------
# Load the training and test datasets
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

y_new= y-1
num_classes = len(np.unique(y_train))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_new, test_size=0.2, random_state=42)

# One-hot encode the target
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

def split_weights_and_biases(weights, biases):
    # Process weights
    positive_weights = np.where(weights > 0, weights, 0)  # Keep only positive weights
    negative_weights = np.where(weights < 0, -weights, 0)  # Absolute values of negative weights

    # Process biases
    positive_biases = np.where(biases > 0, biases, 0)  # Keep only positive biases
    negative_biases = np.where(biases < 0, -biases, 0)  # Absolute values of negative biases

    return positive_weights, negative_weights, positive_biases, negative_biases
# Split into positive and negative parts
def process_model_layers(model):
    results = {}
    for i, layer in enumerate(model.layers):
        # Get weights and biases
        weights = layer.get_weights()
        if weights:
            # Extract weights and biases if available
            layer_weights = weights[0] if len(weights) > 0 else None
            layer_biases = weights[1] if len(weights) > 1 else None

            # Split weights and biases
            if layer_weights is not None and layer_biases is not None:
                pos_weights, neg_weights, pos_biases, neg_biases = split_weights_and_biases(
                    layer_weights, layer_biases
                )

                # Save results
                results[f"Layer_{i}"] = {
                    "Positive_Weights": pos_weights,
                    "Negative_Weights": neg_weights,
                    "Positive_Biases": pos_biases,
                    "Negative_Biases": neg_biases,
                }
    return results
model=load_model(model_path,custom_objects={"clipped_relu": clipped_relu})
layer_results = process_model_layers(model)


w_offset = 0.03125 * 40
b_offset = 0.03125 * 2
gain=  120000
gain_b=gain/10

layer_0_w_p = (layer_results['Layer_0']['Positive_Weights']+w_offset)/gain
layer_0_w_n = (layer_results['Layer_0']['Negative_Weights']+w_offset)/gain
layer_1_w_p = (layer_results['Layer_1']['Positive_Weights']+w_offset)/gain
layer_1_w_n = (layer_results['Layer_1']['Negative_Weights']+w_offset)/gain
layer_2_w_p = (layer_results['Layer_2']['Positive_Weights']+w_offset)/gain
layer_2_w_n = (layer_results['Layer_2']['Negative_Weights']+w_offset)/gain

layer_0_b_p = (layer_results['Layer_0']['Positive_Biases']+b_offset)/gain_b
layer_0_b_n = (layer_results['Layer_0']['Negative_Biases']+b_offset)/gain_b
layer_1_b_p = (layer_results['Layer_1']['Positive_Biases']+b_offset)/gain_b
layer_1_b_n = (layer_results['Layer_1']['Negative_Biases']+b_offset)/gain_b
layer_2_b_p = (layer_results['Layer_2']['Positive_Biases']+b_offset)/gain_b
layer_2_b_n = (layer_results['Layer_2']['Negative_Biases']+b_offset)/gain_b


# print("#********************** weights ***********************")
# print( 1/np.max(layer_0_w_p), 1/np.min(layer_0_w_p))
# print(1/np.max(layer_0_w_n), 1/np.min(layer_0_w_n))
# print(1/np.max(layer_1_w_p) ,1/np.min(layer_1_w_p))
# print(1/np.max(layer_1_w_n),1/np.min(layer_1_w_n))
# print(1/np.max(layer_2_w_p) ,1/np.min(layer_2_w_p))
# print(1/np.max(layer_2_w_n),1/np.min(layer_2_w_n))
# print("#********************** Bias ***********************")
# print( 1/np.max(layer_0_b_p), 1/np.min(layer_0_b_p))
# print(1/np.max(layer_0_b_n), 1/np.min(layer_0_b_n))
# print(1/np.max(layer_1_b_p) ,1/np.min(layer_1_b_p))
# print(1/np.max(layer_1_b_n),1/np.min(layer_1_b_n))
# print(1/np.max(layer_2_b_p) ,1/np.min(layer_2_b_p))
# print(1/np.max(layer_2_b_n),1/np.min(layer_2_b_n))

layer0_weights_p = np.concatenate((layer_0_w_p, layer_0_b_p.reshape(1,64)), axis=0)
layer0_weights_n = np.concatenate((layer_0_w_n, layer_0_b_n.reshape(1,64)), axis=0)
layer1_weights_p = np.concatenate((layer_1_w_p, layer_1_b_p.reshape(1,32)), axis=0)
layer1_weights_n = np.concatenate((layer_1_w_n, layer_1_b_n.reshape(1,32)), axis=0)
layer2_weights_p = np.concatenate((layer_2_w_p, layer_2_b_p.reshape(1,6)), axis=0)
layer2_weights_n = np.concatenate((layer_2_w_n, layer_2_b_n.reshape(1,6)), axis=0)

scaler = gain
# --- Input ---
input_original = Input(shape=(561,), name='input_original')
#lambda_layer = Lambda(lambda x: tf.concat([x, tf.ones_like(x[:, :1]) * 0.1], axis=1))(input_original)
lambda_layer = Lambda(lambda x: tf.concat([x, tf.ones_like(x[:, :1]) * 0.1], axis=1),output_shape=(562,))(input_original)
# --- Define Layers ---
positive_layer0_d = Dense(64, activation=None, use_bias=False, name='layer0_pos_matrix')
negative_layer0_d = Dense(64, activation=None, use_bias=False, name='layer0_neg_matrix')
positive_out0 = positive_layer0_d(lambda_layer)
negative_out0 = negative_layer0_d(lambda_layer)
# --- Subtract and scale ---
combined_layer0 = Subtract()([positive_out0, negative_out0])
# lambda_layer0 = Lambda(lambda x: x * scaler)(combined_layer0)
lambda_layer0 = Lambda(lambda x: x * scaler, output_shape=(64,))(combined_layer0)
activation1 = Activation(clipped_relu)(lambda_layer0)

# --- Second Layer ---

lambda_layer1 = Lambda(lambda x: tf.concat([x, tf.ones_like(x[:, :1]) * 0.1], axis=1),output_shape=(65,))(activation1)
# --- Apply Layers ---
positive_layer1_d = Dense(32, activation=None, use_bias=False, name='layer1_pos_matrix')
negative_layer1_d = Dense(32, activation=None, use_bias=False, name='layer1_neg_matrix')
positive_out1 = positive_layer1_d(lambda_layer1)
negative_out1 = negative_layer1_d(lambda_layer1)
# --- Subtract and scale ---
combined_layer1 = Subtract()([positive_out1, negative_out1])
# lambda_layer0 = Lambda(lambda x: x * scaler)(combined_layer0)
lambda_layer2 = Lambda(lambda x: x * scaler, output_shape=(32,))(combined_layer1)
activation2 = Activation(clipped_relu)(lambda_layer2)

# --- Second Layer ---
lambda_layer3 = Lambda(lambda x: tf.concat([x, tf.ones_like(x[:, :1]) * 0.1], axis=1),output_shape=(33,))(activation2)
# --- Apply Layers ---
positive_layer2_d = Dense(6, activation=None, use_bias=False, name='layer2_pos_matrix')
negative_layer2_d = Dense(6, activation=None, use_bias=False, name='layer2_neg_matrix')
positive_out2 = positive_layer2_d(lambda_layer3)
negative_out2 = negative_layer2_d(lambda_layer3)
combined_layer2 = Subtract()([positive_out2, negative_out2])
# --- Subtract and scale ---
lambda_layer4 = Lambda(lambda x: x * scaler, output_shape=(6,))(combined_layer2)
outputs = Activation('sigmoid')(lambda_layer4)

# --- Build Model ---
new_model = Model(inputs=input_original, outputs=outputs)

#--- Assign weights correctly ---
new_model.get_layer("layer0_pos_matrix").set_weights([layer0_weights_p])
new_model.get_layer("layer0_neg_matrix").set_weights([layer0_weights_n])
new_model.get_layer("layer1_pos_matrix").set_weights([layer1_weights_p])
new_model.get_layer("layer1_neg_matrix").set_weights([layer1_weights_n])
new_model.get_layer("layer2_pos_matrix").set_weights([layer2_weights_p])
new_model.get_layer("layer2_neg_matrix").set_weights([layer2_weights_n])



new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# print(new_model.summary())
loss, accuracy = new_model.evaluate(X_test, y_test_cat)
print(f"Test Accuracy: {accuracy:.4f}")

print(f"Loss : {loss:.4f}")
new_model.save("best_hardware_simulation_model.h5")
df_layer0_p = pd.DataFrame(layer0_weights_p)
df_layer0_n = pd.DataFrame(layer0_weights_n)
df_layer1_p = pd.DataFrame(layer1_weights_p)
df_layer1_n = pd.DataFrame(layer1_weights_n)
df_layer2_p = pd.DataFrame(layer2_weights_p)
df_layer2_n = pd.DataFrame(layer2_weights_n)

# Save each DataFrame to a separate CSV file
df_layer0_p.to_csv("Q_new_models/layer0_weights_pos.csv", index=False)
df_layer0_n.to_csv("Q_new_models/layer0_weights_neg.csv", index=False)
df_layer1_p.to_csv("Q_new_models/layer1_weights_pos.csv", index=False)
df_layer1_n.to_csv("Q_new_models/layer1_weights_neg.csv", index=False)
df_layer2_p.to_csv("Q_new_models/layer2_weights_pos.csv", index=False)
df_layer2_n.to_csv("Q_new_models/layer2_weights_neg.csv", index=False)

layer0_resistance_pos = 1/ np.unique(layer0_weights_p)
y_pred_probs = new_model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_test_cat, axis=1)

# Step 2: Generate confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)

# Step 3: Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

